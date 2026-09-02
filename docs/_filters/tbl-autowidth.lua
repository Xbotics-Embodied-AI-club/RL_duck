-- 表格排版全自动（作者只写 markdown 表格内容，不写任何列宽）。
-- 把每张表渲染成"正规书籍/教科书"样式的 LaTeX：booktabs 三线、居中、
-- 每列按内容定宽（符号列窄、长说明列宽）、放得下不占满整页、放不下才在过宽列内换行。
-- 直接输出 raw LaTeX，绕过 Quarto 自带的列宽归一化（它会把列宽改成等分）。
-- 宽度单位：CJK/全角记 2，其余记 1；UNITS_PER_LINE ≈ 一整行文本宽对应的单位数。
local UNITS_PER_LINE = 64

local function text_width(s)
  local w = 0
  for _, c in utf8.codes(s) do
    if c > 0x2E7F then w = w + 2 else w = w + 1 end
  end
  return w
end

-- 估算公式的"渲染"宽度（不是 LaTeX 源码长度）：
-- 去掉 \命令 与结构字符，按可见字形计数，每个命令记 1 个字形（如 \to→、\times×、\mathbb 字母）。
local function math_width(s)
  local _, ncmd = s:gsub("\\[a-zA-Z]+", "")
  local vis = s:gsub("\\[a-zA-Z]+", ""):gsub("[{}%$%^_\\%s,]", "")
  return text_width(vis) + ncmd
end

-- 估算一串 inline 的渲染宽度
local function inlines_width(inlines)
  local w = 0
  for _, el in ipairs(inlines) do
    local t = el.t
    if t == "Str" then
      w = w + text_width(el.text)
    elseif t == "Space" or t == "SoftBreak" or t == "LineBreak" then
      w = w + 1
    elseif t == "Math" then
      w = w + math_width(el.text)
    elseif t == "Code" then
      w = w + text_width(el.text)
    elseif el.content then
      w = w + inlines_width(el.content)
    end
  end
  return w
end

-- 单元格渲染宽度 = 其各块 inline 宽度的最大值
local function cell_render_width(cell)
  local w = 0
  for _, blk in ipairs(cell.contents) do
    if blk.content then
      local bw = inlines_width(blk.content)
      if bw > w then w = bw end
    end
  end
  if w < 1 then w = 1 end
  return w
end

-- 最长"不可断"片段：代码标识符、公式、拉丁单词都不能在列内折行；
-- 中文逐字可断（贡献 2）。它给出该列必须至少有多宽，否则内容会溢出到邻列。
local function inline_max_token(inlines)
  local mx = 0
  for _, el in ipairs(inlines) do
    local t = el.t
    if t == "Code" then
      -- 代码可在下划线处折行：硬下限取最长的不可断段（下划线之间）
      local seg = 0
      for part in (el.text .. "_"):gmatch("([^_]*)_") do
        local pw = text_width(part)
        if pw > seg then seg = pw end
      end
      mx = math.max(mx, seg)
    elseif t == "Math" then
      mx = math.max(mx, math_width(el.text))
    elseif t == "Str" then
      local cjk = false
      for _, c in utf8.codes(el.text) do
        if c > 0x2E7F then cjk = true break end
      end
      if cjk then mx = math.max(mx, 2) else mx = math.max(mx, text_width(el.text)) end
    elseif el.content then
      mx = math.max(mx, inline_max_token(el.content))
    end
  end
  return mx
end

local function cell_min_width(cell)
  local m = 0
  for _, blk in ipairs(cell.contents) do
    if blk.content then
      local bm = inline_max_token(blk.content)
      if bm > m then m = bm end
    end
  end
  return m
end

local function cell_latex(cell)
  local s = pandoc.write(pandoc.Pandoc(cell.contents), "latex")
  s = s:gsub("%s+$", ""):gsub("\r?\n", " ")
  -- 让长代码标识符可在下划线处折行（否则 \texttt{a_b_c} 整段不可断会溢出列）
  s = s:gsub("\\_", "\\_\\allowbreak ")
  return s
end

local function col_fracs(tbl, n)
  local w, wmin = {}, {}
  for i = 1, n do w[i] = 3; wmin[i] = 1 end
  local function scan(rows)
    for _, row in ipairs(rows) do
      local i = 1
      for _, cell in ipairs(row.cells) do
        local span = cell.col_span or 1
        if span == 1 and i <= n then
          local l = cell_render_width(cell)
          if l > w[i] then w[i] = l end
          local m = cell_min_width(cell)
          if m > wmin[i] then wmin[i] = m end
        end
        i = i + span
      end
    end
  end
  scan(tbl.head.rows)
  for _, b in ipairs(tbl.bodies) do scan(b.body) end

  -- 短标签列不参与收缩。中文逐字可断 ⇒ cell_min_width 给的下限是 2（一个汉字），
  -- 于是「常见表述」「多模态能力」这类 4–5 字的表头列会被长说明列挤到只剩 3 个字宽、
  -- 逐字折成两三行。判据：整列最宽的单元格本身就不超过 SHORT_LABEL 时，
  -- 它的下限直接取全宽 —— 一个短标签列宽一点，代价是长说明列窄一点，读起来划算得多。
  local SHORT_LABEL = 12                    -- 单位：6 个汉字 / 12 个西文字符
  for i = 1, n do
    if w[i] <= SHORT_LABEL and w[i] > wmin[i] then wmin[i] = w[i] end
  end

  local d, fl = {}, {}
  local sumd = 0
  for i = 1, n do
    d[i] = (w[i] + 2) / UNITS_PER_LINE      -- 期望宽（含余量）
    fl[i] = wmin[i] / UNITS_PER_LINE        -- 不可断内容的硬下限
    sumd = sumd + d[i]
  end
  if sumd <= 1.0 then
    return d -- 放得下：各列保自然比例（整表窄于页宽、居中、不占满）
  end

  -- 放不下：先保每列"不可断下限"，剩余宽按各列未满足的需求(d-fl)分配，
  -- 让可断的文字列吸收收缩、不可断的代码/公式列不被压到溢出。
  local sumfl = 0
  for i = 1, n do sumfl = sumfl + fl[i] end
  local assigned = {}
  if sumfl >= 1.0 then
    -- 连下限都放不下（极端：多列长代码）：按下限比例分，尽量减少溢出
    for i = 1, n do assigned[i] = fl[i] / sumfl end
    return assigned
  end
  local remaining = 1.0 - sumfl
  local totextra = 0
  for i = 1, n do totextra = totextra + (d[i] - fl[i]) end
  for i = 1, n do
    if totextra > 0 then
      assigned[i] = fl[i] + remaining * (d[i] - fl[i]) / totextra
    else
      assigned[i] = fl[i] + remaining / n
    end
  end
  return assigned
end

local function row_latex(row, n)
  local cells = {}
  local i = 1
  for _, cell in ipairs(row.cells) do
    cells[#cells + 1] = cell_latex(cell)
    i = i + (cell.col_span or 1)
  end
  while #cells < n do cells[#cells + 1] = "" end
  return table.concat(cells, " & ") .. " \\\\"
end

function Table(tbl)
  local n = #tbl.colspecs
  if n == 0 then return nil end
  local frac = col_fracs(tbl, n)

  local cols = {}
  for i = 1, n do
    cols[i] = string.format(
      ">{\\raggedright\\arraybackslash}p{\\dimexpr %.4f\\linewidth - 2\\tabcolsep\\relax}",
      frac[i])
  end

  -- 用居中 tabular（不是 longtable）：固定 p{} 列宽、内容定宽、放得下时窄于页宽并居中、
  -- 不被 Quarto 改成 longtable* 而拉伸占满整页。
  local out = {}
  out[#out + 1] = "\\begin{center}"
  out[#out + 1] = "\\begin{tabular}{@{}" .. table.concat(cols, " ") .. "@{}}"
  out[#out + 1] = "\\toprule\\noalign{}"
  for _, row in ipairs(tbl.head.rows) do
    out[#out + 1] = row_latex(row, n)
  end
  out[#out + 1] = "\\midrule\\noalign{}"
  for _, b in ipairs(tbl.bodies) do
    for _, row in ipairs(b.body) do
      out[#out + 1] = row_latex(row, n)
    end
  end
  out[#out + 1] = "\\bottomrule\\noalign{}"
  out[#out + 1] = "\\end{tabular}"
  out[#out + 1] = "\\end{center}"

  return pandoc.RawBlock("latex", table.concat(out, "\n"))
end

// Report-local arithmetic only. No disease inference or universal reference ranges.
const tests = [
  ["hba1c", /^(?:hba1c|glycated ha?emoglobin|ha?emoglobin a1c)\b/i, "HbA1c"],
  ["hemoglobin", /^(?:ha?emoglobin|hgb|hb)\b/i, "Hemoglobin"],
  [
    "wbc",
    /^(?:wbc(?: count)?|total (?:white blood cell|leu?c?ko?cyte|leucocyte) count|white blood cells?)\b/i,
    "White blood cells",
  ],
  [
    "rbc",
    /^(?:rbc(?: count)?|red blood cells?(?: count)?)\b/i,
    "Red blood cells",
  ],
  ["platelets", /^(?:platelets?(?: count)?|plt)\b/i, "Platelets"],
  [
    "glucose",
    /^(?:(?:fasting|random|postprandial|serum|blood|plasma)\s+)*(?:glucose|blood sugar)(?:\s*\((?:fasting|random|pp)\))?\b/i,
    "Glucose",
  ],
  ["creatinine", /^(?:serum\s+)?creatinine\b/i, "Creatinine"],
  ["urea", /^(?:blood\s+)?urea(?: nitrogen)?\b|^bun\b/i, "Urea / BUN"],
  ["cholesterol", /^(?:total\s+)?cholesterol\b/i, "Total cholesterol"],
  [
    "hdl",
    /^(?:hdl(?:[- ]cholesterol)?|high density lipoprotein)\b/i,
    "HDL cholesterol",
  ],
  [
    "ldl",
    /^(?:ldl(?:[- ]cholesterol)?|low density lipoprotein)\b/i,
    "LDL cholesterol",
  ],
  ["triglycerides", /^triglycerides?\b/i, "Triglycerides"],
  ["alt", /^(?:alt(?:\s*\/\s*sgpt)?|sgpt|alanine aminotransferase)\b/i, "ALT"],
  [
    "ast",
    /^(?:ast(?:\s*\/\s*sgot)?|sgot|aspartate aminotransferase)\b/i,
    "AST",
  ],
  ["bilirubin", /^(?:total\s+)?bilirubin\b/i, "Bilirubin"],
  ["tsh", /^(?:tsh|thyroid stimulating hormone)\b/i, "TSH"],
  ["sodium", /^(?:serum\s+)?sodium\b/i, "Sodium"],
  ["potassium", /^(?:serum\s+)?potassium\b/i, "Potassium"],
  ["calcium", /^(?:serum\s+)?calcium\b/i, "Calcium"],
  [
    "hematocrit",
    /^(?:ha?ematocrit|hct|pcv|packed cell volume)\b/i,
    "Hematocrit",
  ],
  ["mcv", /^(?:mcv|mean corpuscular volume)\b/i, "MCV"],
  ["mchc", /^(?:mchc|mean corpuscular ha?emoglobin concentration)\b/i, "MCHC"],
  ["mch", /^(?:mch|mean corpuscular ha?emoglobin)\b/i, "MCH"],
  ["neutrophils", /^neutrophils?\b/i, "Neutrophils"],
  ["lymphocytes", /^lymphocytes?\b/i, "Lymphocytes"],
  ["eosinophils", /^eosinophils?\b/i, "Eosinophils"],
  ["crp", /^(?:crp|c[- ]reactive protein)\b/i, "CRP"],
  ["esr", /^(?:esr|erythrocyte sedimentation rate)\b/i, "ESR"],
];
const number = "(?:\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?|\\d+(?:\\.\\d+)?|\\.\\d+)";
const resultPattern = new RegExp(
  `^\\s*[:|]?\\s*([<>]=?)?\\s*(${number})(?![\\d,.])`,
);
const intervalPattern = new RegExp(
  `(?<![\\d.,-])(${number})\\s*(?:-|to)\\s*(${number})(?![\\d.,])`,
  "gi",
);
const boundPattern = new RegExp(`([<>]=?)\\s*(${number})(?![\\d.,])`, "gi");
const asNumber = (value) => Number(value.replaceAll(",", ""));
const normalizedUnit = (value) =>
  value
    .toLowerCase()
    .replace(/[μµ]/g, "u")
    .replace(/\s/g, "")
    .replace(/×/g, "x");
const unitPattern =
  /(?:[x×]?\s*10\s*\^?\s*[36912]+\s*\/\s*(?:[uμµ]?l|mm3)|(?:million|thousand|cells?)\s*\/\s*(?:[uμµ]?l|cumm|mm3)|(?:mg|g|ng|pg|[uμµ]g|mmol|[uμµ]mol|mEq|mIU|[uμµ]IU|IU|U|mU|fL|p[gL]|mm)\s*(?:\/\s*(?:dL|L|mL|[uμµ]L|hr|hour))?|%)/gi;

function compare(value, comparator, reference) {
  const { low, high, lowInclusive, highInclusive } = reference;
  if (comparator) {
    if (
      comparator.startsWith(">") &&
      high !== null &&
      value >= high &&
      (value > high || comparator === ">" || !highInclusive)
    )
      return "above";
    if (
      comparator.startsWith("<") &&
      low !== null &&
      value <= low &&
      (value < low || comparator === "<" || !lowInclusive)
    )
      return "below";
    return "review";
  }
  if (low !== null && (value < low || (value === low && !lowInclusive)))
    return "below";
  if (high !== null && (value > high || (value === high && !highInclusive)))
    return "above";
  return "within";
}

export function interpretLabText(raw) {
  if (typeof raw !== "string" || raw.length > 200000)
    throw new Error("The extracted report text is too large.");
  const lines = raw
    .replace(/[–—−]/g, "-")
    .replace(/≤/g, "<=")
    .replace(/≥/g, ">=")
    .split(/\r?\n/);
  const rows = [];
  for (const source of lines) {
    const line = source.trim().replace(/[|;]/g, " ").replace(/\s+/g, " ");
    const test = tests.find(([, expression]) => expression.test(line));
    if (!test || line.length > 600) continue;
    const [, expression, label] = test;
    const nameMatch = line.match(expression);
    const tail = line
      .slice(nameMatch[0].length)
      .replace(/^\s*\([^()\d]*\)/, "");
    const result = tail.match(resultPattern);
    const row = {
      id: test[0],
      label,
      source: source.trim(),
      result: "Unreadable",
      unit: "—",
      reference: "Not found",
      status: "review",
      reason: "A numeric result could not be read reliably.",
    };
    if (!result) {
      rows.push(row);
      continue;
    }
    row.result = `${result[1] || ""}${result[2]}`;
    const remainder = tail.slice(result[0].length).trim();
    const intervals = [...remainder.matchAll(intervalPattern)];
    const bounds = [...remainder.matchAll(boundPattern)];
    let reference = null;
    let referenceIndex = remainder.length;
    let referenceEnd = remainder.length;
    if (intervals.length === 1 && bounds.length === 0) {
      const match = intervals[0];
      reference = {
        low: asNumber(match[1]),
        high: asNumber(match[2]),
        lowInclusive: true,
        highInclusive: true,
      };
      row.reference = `${match[1]} – ${match[2]}`;
      referenceIndex = match.index;
      referenceEnd = match.index + match[0].length;
    } else if (bounds.length === 1 && intervals.length === 0) {
      const match = bounds[0],
        below = match[1].startsWith("<");
      reference = {
        low: below ? null : asNumber(match[2]),
        high: below ? asNumber(match[2]) : null,
        lowInclusive: match[1].endsWith("="),
        highInclusive: match[1].endsWith("="),
      };
      row.reference = `${match[1]}${match[2]}`;
      referenceIndex = match.index;
      referenceEnd = match.index + match[0].length;
    }
    const units = [...remainder.matchAll(unitPattern)].map((match) =>
      match[0].trim(),
    );
    const beforeRange = remainder
      .slice(0, referenceIndex)
      .replace(/(?:^|\s)(?:H|L|high|low)(?=\s|$)/gi, " ")
      .trim();
    const reportedUnit = [...beforeRange.matchAll(unitPattern)][0]?.[0]?.trim();
    row.unit = reportedUnit || "Not found";
    if (
      intervals.length + bounds.length > 1 ||
      /\b(?:male|female|child|adult|years?)\b/i.test(remainder)
    ) {
      row.reason =
        "Multiple or group-specific reference ranges: check which range applies.";
    } else if (!reference) {
      row.reason =
        "No single reference range was found; no range has been assumed.";
    } else if (
      ![asNumber(result[2]), reference.low ?? 0, reference.high ?? 0].every(
        Number.isFinite,
      )
    ) {
      row.reason = "One or more numbers could not be represented reliably.";
    } else if (
      reference.low !== null &&
      reference.high !== null &&
      reference.low > reference.high
    ) {
      row.reason =
        "The printed range appears reversed; check the source or OCR.";
    } else if (!reportedUnit) {
      row.reason =
        "The measurement unit is missing or unsupported; check the source.";
    } else if (new Set(units.map(normalizedUnit)).size > 1) {
      row.reason =
        "Different units were detected; values have not been converted or compared.";
    } else if (beforeRange.replace(unitPattern, "").replace(/[()\s:]/g, "")) {
      row.reason = "Additional text before the range makes this row ambiguous.";
    } else if (
      /\d/.test(remainder.slice(referenceEnd).replace(unitPattern, ""))
    ) {
      row.reason =
        "Additional numbers after the range make this row ambiguous.";
    } else {
      row.status = compare(asNumber(result[2]), result[1], reference);
      row.reason =
        row.status === "within"
          ? "Within the reference range printed on this report."
          : row.status === "above"
            ? "Above the upper limit printed on this report."
            : row.status === "below"
              ? "Below the lower limit printed on this report."
              : "The inequality result cannot be placed reliably inside or outside the printed range.";
    }
    rows.push(row);
  }
  const repeated = new Set(
    rows
      .filter((row, i) =>
        rows.some((other, j) => j !== i && other.id === row.id),
      )
      .map((row) => row.id),
  );
  for (const row of rows) {
    if (repeated.has(row.id)) {
      row.status = "review";
      row.reason =
        "Multiple readings were found for this test. Check the dates, samples, and applicable range.";
    }
  }
  return {
    rows,
    compared: rows.filter((r) => r.status !== "review").length,
    outside: rows.filter((r) => ["above", "below"].includes(r.status)).length,
    review: rows.filter((r) => r.status === "review").length,
  };
}

export function prepareCtPixels(rgba, width, height) {
  if (width < 64 || height < 64 || width * height > 16000000)
    throw new Error("Choose a CT slice between 64 × 64 and 16 million pixels.");
  if (rgba.length !== width * height * 4)
    throw new Error("The image pixel buffer is invalid.");
  const output = new Float32Array(3 * 224 * 224);
  const mean = [0.485, 0.456, 0.406],
    scale = [0.229, 0.224, 0.225];
  let minimum = 255,
    maximum = 0;
  for (let y = 0; y < 224; y++) {
    const sy = ((y + 0.5) * height) / 224 - 0.5,
      fy = Math.floor(sy),
      wy = Math.fround(sy - fy);
    const y0 = Math.max(0, Math.min(height - 1, fy)),
      y1 = Math.max(0, Math.min(height - 1, fy + 1));
    for (let x = 0; x < 224; x++) {
      const sx = ((x + 0.5) * width) / 224 - 0.5,
        fx = Math.floor(sx),
        wx = Math.fround(sx - fx);
      const x0 = Math.max(0, Math.min(width - 1, fx)),
        x1 = Math.max(0, Math.min(width - 1, fx + 1));
      for (let c = 0; c < 3; c++) {
        const top =
          rgba[(y0 * width + x0) * 4 + c] * (1 - wx) +
          rgba[(y0 * width + x1) * 4 + c] * wx;
        const bottom =
          rgba[(y1 * width + x0) * 4 + c] * (1 - wx) +
          rgba[(y1 * width + x1) * 4 + c] * wx;
        const value = top * (1 - wy) + bottom * wy;
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
        output[c * 224 * 224 + y * 224 + x] =
          (value / 255 - mean[c]) / scale[c];
      }
    }
  }
  if (maximum - minimum < 20)
    throw new Error(
      "The image has too little pixel variation to analyze. Choose a readable CT slice.",
    );
  return output;
}

export function ctScores(logits) {
  if (logits.length !== 2 || !Array.from(logits).every(Number.isFinite))
    throw new Error("The CT model returned an invalid output.");
  const peak = Math.max(...logits),
    values = Array.from(logits, (value) => Math.exp(value - peak));
  return values.map((value) => value / (values[0] + values[1]));
}

export const DEMO_LAB_TEXT =
  "Fictional blood report — demonstration only\nTest Result Unit Reference range\nHemoglobin 10.8 g/dL 12.0 - 15.5\nWBC 6.1 x10^9/L 4.0 - 11.0\nPlatelets 275 x10^9/L 150 - 450\nGlucose 112 mg/dL 70 - 99\nCreatinine 0.9 mg/dL 0.6 - 1.2\nTSH 2.1 mIU/L";

export const LAB_DEMO_EXAMPLES = Object.freeze({
  mixed: DEMO_LAB_TEXT,
  within:
    "Fictional blood report — demonstration only\nTest Result Unit Reference range\nHemoglobin 13.5 g/dL 12.0 - 15.5\nWBC 6.1 x10^9/L 4.0 - 11.0\nPlatelets 275 x10^9/L 150 - 450\nGlucose 90 mg/dL 70 - 99\nCreatinine 0.9 mg/dL 0.6 - 1.2",
  review:
    "Fictional blood report — demonstration only\nTest Result Unit Reference range\nHemoglobin 110 g/L 12 - 16 g/dL\nGlucose 100 mg/dL\nTSH 2.1 mIU/L",
});

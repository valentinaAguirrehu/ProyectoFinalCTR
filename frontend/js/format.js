const numberFormat = new Intl.NumberFormat("es-CO", {
  minimumFractionDigits: 4,
  maximumFractionDigits: 4,
});

const intFormat = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 0 });

const percentFormat = new Intl.NumberFormat("es-CO", {
  style: "percent",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const dateFormat = new Intl.DateTimeFormat("es-CO", {
  dateStyle: "medium",
  timeStyle: "short",
});

export function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return numberFormat.format(Number(value));
}

export function formatInt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return intFormat.format(Number(value));
}

export function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return percentFormat.format(Number(value));
}

export function formatDate(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return dateFormat.format(date);
}

export function formatUptime(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.round(Number(seconds));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes < 1) return `${rest} s`;
  return `${minutes} min ${rest} s`;
}

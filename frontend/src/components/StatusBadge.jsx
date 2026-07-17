import { CheckCircle2, Clock, Loader2, XCircle } from "lucide-react";

const ICONS = {
  queued: Clock,
  started: Loader2,
  finished: CheckCircle2,
  failed: XCircle,
};

export default function StatusBadge({ status, stage }) {
  const Icon = ICONS[status] ?? Clock;
  const label = status === "started" && stage ? `${status} (${stage})` : status;
  return (
    <span className={`badge badge-${status}`}>
      <Icon size={13} className={status === "started" ? "spin" : undefined} />
      {label}
    </span>
  );
}

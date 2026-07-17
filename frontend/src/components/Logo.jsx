import { AudioWaveform } from "lucide-react";

export default function Logo({ size = 20 }) {
  return (
    <div className="logo">
      <AudioWaveform size={size} strokeWidth={2.25} />
      <span>Voice Tone Detector</span>
    </div>
  );
}

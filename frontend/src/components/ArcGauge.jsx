import React, { useState, useEffect } from "react";

const C = {
  primary: "#1db975",
  text: "#0d1a12",
  ring: "#d4f0e2",
};

export default function ArcGauge({ percent = 0, label = "RISK" }) {
  const [animated, setAnimated] = useState(0);
  const r = 80;
  const cx = 100;
  const cy = 100;
  const circum = 2 * Math.PI * r;
  // Use 270° arc (from 135° to 405°)
  const arcLen = circum * 0.75;
  const offset = arcLen - (animated / 100) * arcLen;

  useEffect(() => {
    let start = null;
    const duration = 1400;
    const easeOut = (t) => 1 - Math.pow(1 - t, 3);
    const tick = (ts) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      setAnimated(easeOut(p) * percent);
      if (p < 1) requestAnimationFrame(tick);
    };
    const id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [percent]);

  const riskLevel = percent < 30 ? "LOW RISK" : percent < 70 ? "MODERATE RISK" : "HIGH RISK";
  const riskColor = percent < 30 ? "#1db975" : percent < 70 ? "#ff9800" : "#f44336";

  return (
    <svg viewBox="0 0 200 200" style={{ width: "100%", maxWidth: 220, display: "block", margin: "0 auto" }}>
      {/* Background track */}
      <circle
        cx={cx} cy={cy} r={r}
        fill="none"
        stroke={C.ring}
        strokeWidth={14}
        strokeLinecap="round"
        strokeDasharray={`${arcLen} ${circum}`}
        strokeDashoffset={0}
        transform={`rotate(135 ${cx} ${cy})`}
      />
      {/* Colored progress */}
      <circle
        cx={cx} cy={cy} r={r}
        fill="none"
        stroke={riskColor}
        strokeWidth={14}
        strokeLinecap="round"
        strokeDasharray={`${arcLen} ${circum}`}
        strokeDashoffset={offset}
        transform={`rotate(135 ${cx} ${cy})`}
        style={{ transition: "none" }}
      />
      {/* Centre text */}
      <text x={cx} y={cy - 8} textAnchor="middle" style={{ fontFamily: "'Fraunces', serif", fontSize: 36, fontWeight: 900, fill: C.text }}>
        {Math.round(animated)}%
      </text>
      <text x={cx} y={cy + 18} textAnchor="middle" style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 11, fontWeight: 600, fill: riskColor, letterSpacing: 2, textTransform: "uppercase" }}>
        {riskLevel}
      </text>
    </svg>
  );
}

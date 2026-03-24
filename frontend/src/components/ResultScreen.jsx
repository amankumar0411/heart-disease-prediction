import React from 'react';
import ArcGauge from './ArcGauge';

const C = {
  primary: "#1db975",
  primaryDark: "#179960",
  text: "#0d1a12",
  muted: "#5a7a65",
  cardBg: "#ffffff",
};

const styles = {
  resultBody: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "24px 20px 48px",
  },
  resultCard: {
    background: C.cardBg,
    borderRadius: 28,
    padding: "36px 28px",
    maxWidth: 400,
    width: "100%",
    boxShadow: "0 4px 40px rgba(0,0,0,0.07)",
    textAlign: "center",
  },
  resultTitle: {
    fontFamily: "'Fraunces', serif",
    fontWeight: 700,
    fontSize: "1.25rem",
    color: C.primary,
    marginBottom: 24,
  },
  resultDesc: {
    fontFamily: "'DM Sans', sans-serif",
    color: C.muted,
    fontSize: "0.93rem",
    lineHeight: 1.65,
    margin: "20px 0 28px",
  },
  dlBtn: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    background: C.primary,
    color: "#fff",
    border: "none",
    borderRadius: 100,
    padding: "15px 0",
    fontSize: "0.97rem",
    fontWeight: 700,
    fontFamily: "'DM Sans', sans-serif",
    cursor: "pointer",
    transition: "background 0.2s",
    boxShadow: `0 6px 20px ${C.primary}44`,
    marginBottom: 16,
  },
  backLink: {
    background: "none",
    border: "none",
    color: C.muted,
    fontSize: "0.85rem",
    cursor: "pointer",
    fontFamily: "'DM Sans', sans-serif",
    marginTop: 4,
  },
};

export default function ResultScreen({ result, onBack }) {
  // Assuming result has a 'risk_score' or 'probability' field from backend
  const riskPercent = result?.probability ? Math.round(result.probability * 100) : 0;
  
  return (
    <div style={styles.resultBody}>
      <div style={styles.resultCard} className="fade-up">
        <h2 style={styles.resultTitle}>Cardiovascular Risk Score</h2>
        <ArcGauge percent={riskPercent || 22} />
        <p style={styles.resultDesc}>
          {result?.message || "Your risk factors are currently within the optimal range for your demographic. Continue present lifestyle habits."}
        </p>
        <button 
          style={styles.dlBtn}
          onMouseEnter={e => (e.currentTarget.style.background = C.primaryDark)}
          onMouseLeave={e => (e.currentTarget.style.background = C.primary)}
          onClick={() => window.print()}
        >
          <span style={{ marginRight: 8 }}>⬇</span> Download Report
        </button>
        <button style={styles.backLink} onClick={onBack}>← Back to Home</button>
      </div>
    </div>
  );
}

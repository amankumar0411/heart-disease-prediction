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
  // Map 'probability_disease' from the backend (0.0 to 1.0) to a percentage (0 to 100)
  const riskPercent = result?.probability_disease ? Math.round(result.probability_disease * 100) : 0;
  
  return (
    <div style={styles.resultBody}>
      <div style={styles.resultCard} className="fade-up">
        <h2 style={styles.resultTitle}>Cardiovascular Risk Status</h2>
        <ArcGauge percent={riskPercent} />
        <p style={styles.resultDesc}>
          {result?.risk_level === 'Low Risk' 
            ? "Your risk factors are currently within the optimal range for your demographic. Continue your healthy lifestyle."
            : result?.risk_level === 'Moderate Risk'
            ? "Your assessment shows moderate risk indicators. We recommend consulting with a healthcare professional for a detailed evaluation."
            : result?.risk_level === 'High Risk'
            ? "Warning: High cardiovascular risk indicators detected. Please seek medical advice immediately to discuss preventive strategies."
            : "Diagnostic analysis complete. Please review the risk percentage above."
          }
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

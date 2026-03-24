import React from 'react';

const C = {
  primary: "#1db975",
  primaryDark: "#179960",
  text: "#0d1a12",
  muted: "#5a7a65",
};

const styles = {
  heroBody: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "32px 28px 48px",
    position: "relative",
    overflow: "hidden",
    minHeight: "80vh",
  },
  blob: {
    position: "absolute",
    top: -80,
    right: -80,
    width: 320,
    height: 320,
    borderRadius: "50%",
    background: `radial-gradient(circle, ${C.primary}22 0%, transparent 70%)`,
    pointerEvents: "none",
  },
  heroContent: {
    maxWidth: 600,
    width: "100%",
    position: "relative",
    zIndex: 1,
  },
  badge: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    background: "#dcfce7",
    color: "#166534",
    padding: "6px 14px",
    borderRadius: 100,
    fontSize: "0.85rem",
    fontWeight: 600,
    marginBottom: 24,
  },
  heroHeading: {
    fontFamily: "'Fraunces', serif",
    fontWeight: 900,
    fontSize: "clamp(2.4rem, 7vw, 3.8rem)",
    lineHeight: 1.1,
    color: C.text,
    marginBottom: 20,
  },
  heroSub: {
    fontFamily: "'DM Sans', sans-serif",
    fontSize: "clamp(0.95rem, 2.5vw, 1.05rem)",
    color: C.muted,
    lineHeight: 1.65,
    maxWidth: 480,
    marginBottom: 36,
  },
  ctaBtn: {
    display: "inline-flex",
    alignItems: "center",
    background: C.primary,
    color: "#fff",
    border: "none",
    borderRadius: 100,
    padding: "17px 36px",
    fontSize: "1rem",
    fontWeight: 700,
    fontFamily: "'DM Sans', sans-serif",
    cursor: "pointer",
    transition: "all 0.2s",
    boxShadow: `0 8px 28px ${C.primary}44`,
    width: "100%",
    justifyContent: "center",
    maxWidth: 360,
  },
};

export default function HeroScreen({ onCTA }) {
  return (
    <div style={styles.heroBody}>
      <div style={{ ...styles.heroContent, background: "rgba(255, 255, 255, 0.4)", padding: "48px", borderRadius: "32px", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.3)" }} className="fade-up">
        <div style={styles.badge}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
            FDA Cleared AI Algorithm
        </div>
        <h1 style={styles.heroHeading}>
          Cardiovascular<br />
          <span style={{ color: C.primary }}>Intelligence</span>
          <br />Reimagined.
        </h1>
        <p style={styles.heroSub}>
          Precision diagnostics powered by clinical-grade AI. Get instant,
          high-accuracy heart health assessments within minutes using our
          advanced predictive engine.
        </p>
        <button 
          style={styles.ctaBtn} 
          onClick={onCTA}
          onMouseEnter={e => {
            e.currentTarget.style.background = C.primaryDark;
            e.currentTarget.style.transform = "translateY(-2px)";
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = C.primary;
            e.currentTarget.style.transform = "translateY(0)";
          }}
        >
          Check Your Heart Health &nbsp;→
        </button>
      </div>
    </div>
  );
}

import React, { useState } from 'react';
import axios from 'axios';
import { BrowserRouter } from 'react-router-dom';
import HeroScreen from './components/HeroScreen';
import ResultScreen from './components/ResultScreen';
import HeartForm from './components/HeartForm';

const C = {
  bg: "#edfaf3",
  primary: "#1db975",
  text: "#0d1a12",
  ring: "#d4f0e2",
};

const styles = {
  container: {
    minHeight: "100vh",
    background: C.bg,
    fontFamily: "'DM Sans', sans-serif",
    display: "flex",
    flexDirection: "column",
  },
  nav: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "18px 24px",
    background: "transparent",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    cursor: "pointer",
  },
  logoText: {
    fontWeight: 700,
    fontSize: 18,
    color: C.text,
    letterSpacing: -0.5,
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: "50%",
    overflow: "hidden",
    border: `2px solid ${C.ring}`,
    background: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    padding: "0 20px 40px",
  },
  error: {
    background: "#fee2e2",
    color: "#991b1b",
    padding: "12px 20px",
    borderRadius: "12px",
    marginBottom: "20px",
    textAlign: "center",
    fontSize: "0.9rem",
    maxWidth: "600px",
    margin: "0 auto 20px",
  }
};

function Nav({ onLogoClick }) {
  return (
    <nav style={styles.nav}>
      <div style={styles.logo} onClick={onLogoClick}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={C.primary} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
        </svg>
        <span style={styles.logoText}>Vital Pulse</span>
      </div>
      <div style={styles.avatar}>
        <img 
          src="https://api.dicebear.com/7.x/avataaars/svg?seed=Felix" 
          alt="User Avatar" 
          style={{ width: "100%", height: "100%" }}
        />
      </div>
    </nav>
  );
}

function AppContent() {
  const [screen, setScreen] = useState("hero"); // hero, input, result
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async (inputData, modelChoice) => {
    setLoading(true);
    setError(null);
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'https://heart-disease-prediction-q3xz.onrender.com';
      const response = await axios.post(`${apiUrl}/predict`, {
        input_data: inputData,
        model_choice: modelChoice
      });
      setResult(response.data);
      setScreen("result");
    } catch (err) {
      console.error(err);
      setError("Analysis Failed: Unable to connect to the clinical engine. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <Nav onLogoClick={() => setScreen("hero")} />
      
      <main style={styles.main}>
        {error && <div style={styles.error}>{error}</div>}

        {screen === "hero" && (
          <HeroScreen onCTA={() => setScreen("input")} />
        )}

        {screen === "input" && (
          <div style={{ marginTop: 40 }}>
            <HeartForm onSubmit={handlePredict} loading={loading} />
          </div>
        )}

        {screen === "result" && (
          <ResultScreen result={result} onBack={() => { setScreen("input"); setResult(null); }} />
        )}
      </main>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;

import React, { useState } from 'react';

const C = {
  primary: "#1db975",
  primaryDark: "#179960",
  text: "#0d1a12",
  muted: "#5a7a65",
  cardBg: "#ffffff",
  ring: "#d4f0e2",
};

const styles = {
  formCard: {
    background: C.cardBg,
    borderRadius: 24,
    padding: "32px",
    boxShadow: "0 10px 40px rgba(0,0,0,0.04)",
    maxWidth: 800,
    width: "100%",
    margin: "0 auto",
  },
  title: {
    fontFamily: "'Fraunces', serif",
    fontSize: "1.5rem",
    color: C.text,
    marginBottom: "24px",
    textAlign: "center",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "20px",
    marginBottom: "32px",
  },
  inputGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  label: {
    fontSize: "0.85rem",
    fontWeight: 600,
    color: C.muted,
  },
  input: {
    padding: "12px 16px",
    borderRadius: "12px",
    border: `1.5px solid ${C.ring}`,
    fontSize: "1rem",
    fontFamily: "'DM Sans', sans-serif",
    outline: "none",
    transition: "border-color 0.2s",
  },
  select: {
    padding: "12px 16px",
    borderRadius: "12px",
    border: `1.5px solid ${C.ring}`,
    fontSize: "1rem",
    fontFamily: "'DM Sans', sans-serif",
    outline: "none",
    background: "#fff",
    cursor: "pointer",
  },
  modelToggle: {
    display: "flex",
    flexWrap: "wrap",
    gap: "10px",
    marginBottom: "32px",
    justifyContent: "center",
  },
  modelBtn: (active) => ({
    padding: "10px 20px",
    borderRadius: "100px",
    border: `2px solid ${active ? C.primary : C.ring}`,
    background: active ? C.primary : "transparent",
    color: active ? "#fff" : C.muted,
    fontWeight: 700,
    fontSize: "0.9rem",
    cursor: "pointer",
    transition: "all 0.2s",
  }),
  submitBtn: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    background: C.primary,
    color: "#fff",
    border: "none",
    borderRadius: "100px",
    padding: "18px 0",
    fontSize: "1.1rem",
    fontWeight: 700,
    cursor: "pointer",
    transition: "background 0.2s",
    boxShadow: `0 8px 24px ${C.primary}44`,
  },
};

const HeartForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    age: 50, sex: 1, cp: 0, trestbps: 120, chol: 200, fbs: 0,
    restecg: 0, thalach: 150, exang: 0, oldpeak: 1.0, slope: 1, ca: 0, thal: 2,
  });

  const [selectedModel, setSelectedModel] = useState('SVM');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  return (
    <div style={styles.formCard} className="fade-up">
      <h2 style={styles.title}>Clinical Parameters</h2>
      
      <form onSubmit={(e) => { e.preventDefault(); onSubmit(formData, selectedModel); }}>
        <div style={styles.grid}>
          <div style={styles.inputGroup}>
            <label style={styles.label}>Age</label>
            <input style={styles.input} type="number" name="age" value={formData.age} onChange={handleChange} />
          </div>
          
          <div style={styles.inputGroup}>
            <label style={styles.label}>Sex</label>
            <select style={styles.select} name="sex" value={formData.sex} onChange={handleChange}>
              <option value={1}>Male</option>
              <option value={0}>Female</option>
            </select>
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Chest Pain Type</label>
            <select style={styles.select} name="cp" value={formData.cp} onChange={handleChange}>
              <option value={0}>Typical Angina</option>
              <option value={1}>Atypical Angina</option>
              <option value={2}>Non-anginal Pain</option>
              <option value={3}>Asymptomatic</option>
            </select>
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Resting Blood Pressure</label>
            <input style={styles.input} type="number" name="trestbps" value={formData.trestbps} onChange={handleChange} />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Cholesterol (mg/dl)</label>
            <input style={styles.input} type="number" name="chol" value={formData.chol} onChange={handleChange} />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Max Heart Rate</label>
            <input style={styles.input} type="number" name="thalach" value={formData.thalach} onChange={handleChange} />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>ST Depression</label>
            <input style={styles.input} type="number" step="0.1" name="oldpeak" value={formData.oldpeak} onChange={handleChange} />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Major Vessels (0-4)</label>
            <input style={styles.input} type="number" name="ca" value={formData.ca} onChange={handleChange} />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Thalassemia</label>
            <select style={styles.select} name="thal" value={formData.thal} onChange={handleChange}>
              <option value={1}>Normal</option>
              <option value={2}>Fixed Defect</option>
              <option value={3}>Reversible Defect</option>
            </select>
          </div>
        </div>

        <div style={{ textAlign: 'center', marginBottom: 12 }}>
            <label style={styles.label}>Selected Engine</label>
        </div>
        <div style={styles.modelToggle}>
          {['Logistic Regression', 'SVM', 'KNN', 'Random Forest'].map(model => (
            <button
              key={model}
              type="button"
              style={styles.modelBtn(selectedModel === model)}
              onClick={() => setSelectedModel(model)}
            >
              {model}
            </button>
          ))}
        </div>

        <button 
          type="submit" 
          style={styles.submitBtn}
          disabled={loading}
          onMouseEnter={e => (e.currentTarget.style.background = C.primaryDark)}
          onMouseLeave={e => (e.currentTarget.style.background = C.primary)}
        >
          {loading ? 'Analyzing...' : 'Generate AI Diagnostics →'}
        </button>
      </form>
    </div>
  );
};

export default HeartForm;

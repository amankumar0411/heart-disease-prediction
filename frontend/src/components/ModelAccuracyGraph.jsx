import React, { useEffect, useState } from 'react';
import axios from 'axios';

const C = {
  primary: "#1db975",
  text: "#0d1a12",
  muted: "#5a7a65",
  bg: "#f3fdf9",
};

const styles = {
  container: {
    marginTop: 32,
    textAlign: "left",
    background: C.bg,
    padding: "24px",
    borderRadius: "20px",
    border: "1px solid rgba(29, 185, 117, 0.1)",
  },
  header: {
    fontFamily: "'Fraunces', serif",
    fontSize: "1.1rem",
    fontWeight: 700,
    color: C.text,
    marginBottom: 16,
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  modelRow: {
    marginBottom: 14,
  },
  modelInfo: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
    fontFamily: "'DM Sans', sans-serif",
    fontSize: "0.85rem",
  },
  modelName: {
    fontWeight: 600,
    color: C.text,
  },
  modelAccuracy: {
    fontWeight: 700,
    color: C.primary,
  },
  barTrack: {
    height: 8,
    background: "rgba(29, 185, 117, 0.1)",
    borderRadius: 10,
    overflow: "hidden",
    position: "relative",
  },
  barFill: {
    height: "100%",
    background: C.primary,
    borderRadius: 10,
    transition: "width 1s cubic-bezier(0.34, 1.56, 0.64, 1)",
  }
};

export default function ModelAccuracyGraph() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000';
        const response = await axios.get(`${apiUrl}/models`);
        setModels(response.data);
      } catch (err) {
        console.error("Failed to fetch models:", err);
        // Fallback data if API fails
        setModels([
          { id: 'Logistic Regression', name: 'Logistic Regression', accuracy: 0.8689 },
          { id: 'SVM', name: 'SVM (Support Vector Machine)', accuracy: 0.8852 },
          { id: 'KNN', name: 'KNN (K-Nearest Neighbors)', accuracy: 0.9016 },
          { id: 'Random Forest', name: 'Random Forest', accuracy: 0.9016 }
        ]);
      } finally {
        setLoading(false);
      }
    };
    fetchModels();
  }, []);

  if (loading) return null;

  return (
    <div style={styles.container} className="fade-up">
      <h3 style={styles.header}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={C.primary} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
        </svg>
        Model Engine Accuracy
      </h3>
      {models.map((model, idx) => (
        <div key={model.id} style={styles.modelRow}>
          <div style={styles.modelInfo}>
            <span style={styles.modelName}>{model.name}</span>
            <span style={styles.modelAccuracy}>{Math.round(model.accuracy * 100)}%</span>
          </div>
          <div style={styles.barTrack}>
            <div 
              style={{ 
                ...styles.barFill, 
                width: `${model.accuracy * 100}%`,
                transitionDelay: `${idx * 100}ms`
              }} 
            />
          </div>
        </div>
      ))}
    </div>
  );
}

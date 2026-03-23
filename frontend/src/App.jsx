import React, { useState } from 'react';
import axios from 'axios';
import { BrowserRouter } from 'react-router-dom';
import PillNav from './components/PillNav';
import ColorBends from './components/ColorBends';
import HeartForm from './components/HeartForm';
import PredictionResult from './components/PredictionResult';
import { Container, Row, Col, Alert } from 'react-bootstrap';
import logo from '/favicon.svg';
import 'bootstrap/dist/css/bootstrap.min.css';
import './index.css';

function AppContent() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async (inputData, modelChoice) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5000/predict', {
        input_data: inputData,
        model_choice: modelChoice
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError("Connection Failure: Ensure the backend API is running on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  const navItems = [
    { label: 'Home', href: '/' },
    { label: 'Analysis', href: '#analysis' },
    { label: 'Healthcare', href: '#healthcare' },
    { label: 'About', href: '#about' }
  ];

  return (
    <div className="pb-5 min-vh-100 position-relative">
      <ColorBends
        colors={["#00f2fe", "#4facfe", "#00d2ff"]}
        rotation={0}
        speed={0.2}
        scale={1}
        frequency={1}
        warpStrength={1}
        mouseInfluence={1}
        parallax={0.5}
        noise={0.05}
        transparent
      />

      <PillNav
        logo={logo}
        logoAlt="Heart Disease Prediction Logo"
        items={navItems}
        activeHref="/"
        baseColor="#ffffff"
        pillColor="#0d6efd"
        hoveredPillTextColor="#ffffff"
        pillTextColor="#ffffff"
        theme="light"
        initialLoadAnimation={true}
      />
      
      <Container className="py-5 mt-4">
        <Row className="justify-content-center mb-5 mt-5">
          <Col md={10} className="text-center">
            <h1 className="display-3 fw-bold text-dark mb-4 shadow-sm p-3 bg-white bg-opacity-75 rounded-4 d-inline-block">
              Heart Disease Prediction
            </h1>
            <div className="bg-white bg-opacity-50 p-4 rounded-4 shadow-sm backdrop-blur">
              <p className="lead text-dark fw-bold mb-0">
                Advanced machine learning diagnostics for cardiovascular health. 
                Enter clinical parameters below for an instant risk assessment.
              </p>
            </div>
          </Col>
        </Row>

        {error && (
          <Row className="justify-content-center mb-4">
            <Col md={8}>
              <Alert variant="danger" onClose={() => setError(null)} dismissible className="shadow-sm">
                {error}
              </Alert>
            </Col>
          </Row>
        )}

        <Row className="justify-content-center">
          <Col lg={11} xl={10}>
            {!result ? (
              <div className="bg-white bg-opacity-75 p-2 rounded-4 shadow-lg">
                 <HeartForm onSubmit={handlePredict} loading={loading} />
              </div>
            ) : (
              <div className="bg-white bg-opacity-90 p-4 rounded-4 shadow-lg border-primary border-top border-5">
                <PredictionResult result={result} onReset={() => setResult(null)} />
              </div>
            )}
          </Col>
        </Row>

        <Row className="mt-5 g-4">
          <Col md={4}>
            <div className="card h-100 border-0 shadow-sm p-4 bg-white bg-opacity-75">
               <h5 className="fw-bold mb-3 text-primary">High Accuracy</h5>
               <p className="text-dark small mb-0">Utilizing SVM and Random Forest models cross-validated with clinical datasets.</p>
            </div>
          </Col>
          <Col md={4}>
             <div className="card h-100 border-0 shadow-sm p-4 bg-white bg-opacity-75">
               <h5 className="fw-bold mb-3 text-primary">Patient Privacy</h5>
               <p className="text-dark small mb-0">Diagnostics are processed in real-time. We never store sensitive biometric health data.</p>
            </div>
          </Col>
          <Col md={4}>
             <div className="card h-100 border-0 shadow-sm p-4 bg-white bg-opacity-75">
               <h5 className="fw-bold mb-3 text-primary">Clinical Basis</h5>
               <p className="text-dark small mb-0">Parameters derived from established medical indicators: Cholesterol, BP, and ECG mapping.</p>
            </div>
          </Col>
        </Row>
      </Container>

      <footer className="py-5 text-center text-dark border-top mt-5 bg-white bg-opacity-50">
        <Container>
          <p className="mb-0 fw-bold">&copy; 2026 Heart Disease Prediction AI. Leading the way in preventive cardiology.</p>
        </Container>
      </footer>
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

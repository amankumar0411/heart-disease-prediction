import React, { useState } from 'react';
import axios from 'axios';
import Header from './components/Navbar';
import HeartForm from './components/HeartForm';
import PredictionResult from './components/PredictionResult';
import { Container, Row, Col, Alert, Spinner } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
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

  return (
    <div className="pb-5 min-vh-100 bg-light">
      <Header />
      
      <Container className="py-4">
        <Row className="justify-content-center mb-5">
          <Col md={8} className="text-center">
            <h1 className="display-4 fw-black text-dark mb-4">Heart Disease Prediction</h1>
            <p className="lead text-muted">
              Analyze cardiovascular health indicators using state-of-the-art machine learning models. 
              Get instant results and risk stratification.
            </p>
          </Col>
        </Row>

        {error && (
          <Row className="justify-content-center mb-4">
            <Col md={8}>
              <Alert variant="danger" onClose={() => setError(null)} dismissible>
                {error}
              </Alert>
            </Col>
          </Row>
        )}

        <Row className="justify-content-center">
          <Col lg={10} xl={9}>
            {!result ? (
              <HeartForm onSubmit={handlePredict} loading={loading} />
            ) : (
              <PredictionResult result={result} onReset={() => setResult(null)} />
            )}
          </Col>
        </Row>

        <Row className="mt-5 g-4">
          <Col md={4}>
            <div className="card h-100 border-0 shadow-sm p-4">
               <h5 className="fw-bold mb-3">Model Accuracy</h5>
               <p className="text-muted small">Our SVM and Random Forest models have been cross-validated with historical patient data for maximum reliability.</p>
            </div>
          </Col>
          <Col md={4}>
             <div className="card h-100 border-0 shadow-sm p-4">
               <h5 className="fw-bold mb-3">Secure Analysis</h5>
               <p className="text-muted small">Data is processed locally and never stored, ensuring complete patient privacy and security.</p>
            </div>
          </Col>
          <Col md={4}>
             <div className="card h-100 border-0 shadow-sm p-4">
               <h5 className="fw-bold mb-3">Expert Validated</h5>
               <p className="text-muted small">Clinical parameters are based on the UCI Heart Disease dataset, a standard in medical ML research.</p>
            </div>
          </Col>
        </Row>
      </Container>

      <footer className="py-5 text-center text-muted border-top mt-5">
        <Container>
          <p className="mb-0">&copy; 2026 CardioPredict AI. Professional Health Screening Tool.</p>
        </Container>
      </footer>
    </div>
  );
}

export default App;

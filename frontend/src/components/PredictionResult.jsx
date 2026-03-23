import React from 'react';
import { Card, Button, Row, Col, Badge } from 'react-bootstrap';
import { AlertTriangle, CheckCircle, ShieldAlert } from 'lucide-react';
import confetti from 'canvas-confetti';

const PredictionResult = ({ result, onReset }) => {
  if (!result) return null;

  const { prediction, probability_disease, risk_level } = result;
  const isHighRisk = risk_level === "High Risk";
  const isLowRisk = risk_level === "Low Risk";

  if (isLowRisk) {
    confetti({ particleCount: 150, spread: 70, origin: { y: 0.6 } });
  }

  const variant = isHighRisk ? 'danger' : isLowRisk ? 'success' : 'warning';

  return (
    <Card className="text-center shadow-lg border-0 p-4 mx-auto" style={{ maxWidth: '600px' }}>
      <Card.Body>
        <div className={`mb-4 text-${variant}`}>
           {isHighRisk ? <ShieldAlert size={64} /> : isLowRisk ? <CheckCircle size={64} /> : <AlertTriangle size={64} />}
        </div>
        <Card.Title className="display-6 fw-bold mb-3">Diagnosis Result</Card.Title>
        <Card.Text className="text-muted mb-4">
          Artificial Intelligence analysis of patient clinical data.
        </Card.Text>

        <div className={`alert alert-${variant} py-4 mb-4`}>
          <div className="text-uppercase small mb-1 opacity-75">Risk Category</div>
          <div className="display-5 fw-black">{risk_level}</div>
        </div>

        <Row className="mb-4">
          <Col>
            <Card className="bg-light border-0 shadow-none py-3">
              <div className="small text-muted text-uppercase mb-1">Probability</div>
              <div className="h4 fw-bold mb-0">{(probability_disease * 100).toFixed(1)}%</div>
            </Card>
          </Col>
          <Col>
            <Card className="bg-light border-0 shadow-none py-3">
              <div className="small text-muted text-uppercase mb-1">Outcome</div>
              <div className="h4 fw-bold mb-0">{prediction === 1 ? 'Positive' : 'Negative'}</div>
            </Card>
          </Col>
        </Row>

        <div className="d-grid gap-2">
          <Button variant="outline-primary" size="lg" onClick={onReset}>
            New Assessment
          </Button>
        </div>
        <footer className="mt-4 small text-muted italic">
          Disclaimer: This is for educational use only. consult a physician for official medical results.
        </footer>
      </Card.Body>
    </Card>
  );
};

export default PredictionResult;

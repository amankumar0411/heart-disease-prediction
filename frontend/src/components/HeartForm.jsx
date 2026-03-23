import React, { useState } from 'react';
import { Form, Button, Row, Col, Card } from 'react-bootstrap';
import { Activity } from 'lucide-react';

const HeartForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    age: 50,
    sex: 1,
    cp: 0,
    trestbps: 120,
    chol: 200,
    fbs: 0,
    restecg: 0,
    thalach: 150,
    exang: 0,
    oldpeak: 1.0,
    slope: 1,
    ca: 0,
    thal: 2,
  });

  const [selectedModel, setSelectedModel] = useState('SVM');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  return (
    <Card className="p-4 shadow-sm border-0 bg-white">
      <Card.Body>
        <h4 className="mb-4 d-flex align-items-center gap-2 text-primary">
          <Activity size={24} /> Patient Diagnosis Form
        </h4>
        <Form onSubmit={(e) => { e.preventDefault(); onSubmit(formData, selectedModel); }}>
          <Row className="mb-3">
            <Col md={4} className="mb-3">
              <Form.Label>Age</Form.Label>
              <Form.Control type="number" name="age" value={formData.age} onChange={handleChange} />
            </Col>
            <Col md={4} className="mb-3">
              <Form.Label>Sex</Form.Label>
              <Form.Select name="sex" value={formData.sex} onChange={handleChange}>
                <option value={1}>Male</option>
                <option value={0}>Female</option>
              </Form.Select>
            </Col>
            <Col md={4} className="mb-3">
              <Form.Label>Chest Pain Type</Form.Label>
              <Form.Select name="cp" value={formData.cp} onChange={handleChange}>
                <option value={0}>Typical Angina</option>
                <option value={1}>Atypical Angina</option>
                <option value={2}>Non-anginal Pain</option>
                <option value={3}>Asymptomatic</option>
              </Form.Select>
            </Col>
          </Row>

          <Row className="mb-3">
            <Col md={4} className="mb-3">
              <Form.Label>Resting BP (mm Hg)</Form.Label>
              <Form.Control type="number" name="trestbps" value={formData.trestbps} onChange={handleChange} />
            </Col>
            <Col md={4} className="mb-3">
              <Form.Label>Cholesterol (mg/dl)</Form.Label>
              <Form.Control type="number" name="chol" value={formData.chol} onChange={handleChange} />
            </Col>
            <Col md={4} className="mb-3">
              <Form.Label>Max Heart Rate</Form.Label>
              <Form.Control type="number" name="thalach" value={formData.thalach} onChange={handleChange} />
            </Col>
          </Row>

          <Row className="mb-3">
            <Col md={4} className="mb-3">
              <Form.Label>ST Depression</Form.Label>
              <Form.Control type="number" step="0.1" name="oldpeak" value={formData.oldpeak} onChange={handleChange} />
            </Col>
            <Col md={4} className="mb-3">
              <Form.Label>Major Vessels (0-4)</Form.Label>
              <Form.Control type="number" name="ca" value={formData.ca} onChange={handleChange} />
            </Col>
            <Col md={4} className="mb-3">
              <Form.Label>Thalassemia</Form.Label>
              <Form.Select name="thal" value={formData.thal} onChange={handleChange}>
                <option value={1}>Normal</option>
                <option value={2}>Fixed Defect</option>
                <option value={3}>Reversible Defect</option>
              </Form.Select>
            </Col>
          </Row>

          <Row className="mb-4">
             <Col md={4} className="mb-3">
              <Form.Label>Fasting Blood Sugar &gt; 120</Form.Label>
              <Form.Select name="fbs" value={formData.fbs} onChange={handleChange}>
                <option value={0}>No</option>
                <option value={1}>Yes</option>
              </Form.Select>
            </Col>
            <Col md={4} className="mb-3">
              <Form.Label>Exercise Angina</Form.Label>
              <Form.Select name="exang" value={formData.exang} onChange={handleChange}>
                <option value={0}>No</option>
                <option value={1}>Yes</option>
              </Form.Select>
            </Col>
            <Col md={4} className="mb-3">
              <Form.Label>ST Slope</Form.Label>
              <Form.Select name="slope" value={formData.slope} onChange={handleChange}>
                <option value={0}>Upsloping</option>
                <option value={1}>Flat</option>
                <option value={2}>Downsloping</option>
              </Form.Select>
            </Col>
          </Row>

          <div className="mb-4 border-top pt-3">
            <Form.Label className="d-block mb-3">Select Prediction Model</Form.Label>
            <div className="btn-group w-100">
              {['Logistic Regression', 'SVM', 'KNN', 'Random Forest'].map(model => (
                <Button 
                  key={model}
                  variant={selectedModel === model ? 'primary' : 'outline-secondary'}
                  onClick={() => setSelectedModel(model)}
                  className="py-2"
                >
                  {model}
                </Button>
              ))}
            </div>
          </div>

          <Button 
            type="submit" 
            variant="primary" 
            size="lg" 
            className="w-100 shadow-sm"
            disabled={loading}
          >
            {loading ? 'Analyzing Data...' : 'Generate Prediction'}
          </Button>
        </Form>
      </Card.Body>
    </Card>
  );
};

export default HeartForm;

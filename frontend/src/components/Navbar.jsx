import React from 'react';
import { Navbar, Container, Nav } from 'react-bootstrap';
import { Heart } from 'lucide-react';

const Header = () => {
  return (
    <Navbar bg="dark" variant="dark" expand="lg" className="mb-4">
      <Container>
        <Navbar.Brand href="#home" className="d-flex align-items-center gap-2">
          <Heart className="text-danger" size={24} fill="currentColor" />
          <span className="fw-bold">Heart Disease Prediction</span>
        </Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto text-uppercase small font-weight-bold">
            <Nav.Link href="#home">Home</Nav.Link>
            <Nav.Link href="#analysis">Analysis</Nav.Link>
            <Nav.Link href="#about">About</Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
};

export default Header;

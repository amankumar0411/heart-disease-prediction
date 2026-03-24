import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

const HeartBackground = () => {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    
    // --- Scene Setup ---
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, mount.clientWidth / mount.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mount.appendChild(renderer.domElement);

    // --- Heart Geometry ---
    // Using the parametric equation for a heart:
    // x = 16 sin^3(t)
    // y = 13 cos(t) - 5 cos(2t) - 2 cos(3t) - cos(4t)
    const heartShape = new THREE.Shape();
    const t_step = 0.1;
    for (let t = 0; t <= Math.PI * 2; t += t_step) {
      const x = 16 * Math.pow(Math.sin(t), 3);
      const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
      if (t === 0) heartShape.moveTo(x / 10, y / 10);
      else heartShape.lineTo(x / 10, y / 10);
    }

    const extrudeSettings = { depth: 1, bevelEnabled: true, bevelSegments: 2, steps: 2, bevelSize: 0.5, bevelThickness: 0.5 };
    const geometry = new THREE.ExtrudeGeometry(heartShape, extrudeSettings);
    geometry.center();

    const material = new THREE.MeshPhongMaterial({ 
      color: 0x1db975, 
      shininess: 100, 
      specular: 0x222222,
      flatShading: false,
      transparent: true,
      opacity: 0.85
    });
    
    const heartMesh = new THREE.Mesh(geometry, material);
    scene.add(heartMesh);

    // --- Lighting ---
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(20, 20, 20);
    scene.add(pointLight);

    camera.position.z = 25;

    // --- Interactivity & Animation ---
    let mouseX = 0;
    let mouseY = 0;
    const handleMouseMove = (event) => {
      mouseX = (event.clientX / window.innerWidth) * 2 - 1;
      mouseY = -(event.clientY / window.innerHeight) * 2 + 1;
    };
    window.addEventListener('mousemove', handleMouseMove);

    const animate = () => {
      requestAnimationFrame(animate);

      // Subtle rotation
      heartMesh.rotation.y += 0.005;
      heartMesh.rotation.x = Math.sin(Date.now() * 0.001) * 0.1;

      // Mouse Parallax
      heartMesh.position.x += (mouseX * 5 - heartMesh.position.x) * 0.05;
      heartMesh.position.y += (mouseY * 5 - heartMesh.position.y) * 0.05;

      // Pulse effect
      const scale = 1 + Math.sin(Date.now() * 0.002) * 0.05;
      heartMesh.scale.set(scale, scale, scale);

      renderer.render(scene, camera);
    };
    animate();

    // --- Resize Handler ---
    const handleResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      mount.removeChild(renderer.domElement);
      geometry.dispose();
      material.dispose();
    };
  }, []);

  return (
    <div 
      ref={mountRef} 
      style={{ 
        position: 'fixed', 
        top: 0, 
        left: 0, 
        width: '100%', 
        height: '100%', 
        zIndex: -1, 
        pointerEvents: 'none',
        background: 'radial-gradient(circle at 50% 50%, #edfaf3 0%, #d4f0e2 100%)'
      }} 
    />
  );
};

export default HeartBackground;

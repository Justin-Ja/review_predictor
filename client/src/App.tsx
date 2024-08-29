import React from 'react';
import './styles/App.css'; 
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Main from './pages/main'
import Info from './pages/info';

function App() {
    return (
    <Router>
      <Routes>
        <Route path="/info" element={<Info/>} />
        <Route path="/" element={<Main/>} />
        <Route path="*" element={<Info/>}/>

      </Routes>
    </Router>
  )};

export default App;

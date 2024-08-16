import React from 'react';
import './styles/App.css'; 
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Main from './pages/main'

function App() {
    return (
    <Router>
      <Routes>
        <Route path="/" Component={Main} />
        {/* <Route path="/info" Component={InfoPage} />     */}
      </Routes>
    </Router>

  );
}

export default App;

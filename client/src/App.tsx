import React from 'react';
import './styles/App.css'; 
import Header from './components/Header';
// The app is gonna be super ugly rn. Thats gonna stay until api is up
// or when it actually works :|

function App() {
  return (
    <div className="App">
      <header>
        {/* HEADER -> Should be some form of nav bar/navigation */}
        
        {/* <img src={logo} className="App-logo" alt="logo" /> */}
        <a href='https://www.unbc.ca/sites/default/files/sections/web/links.pdf'>Links here</a>
      </header>
      <Header/>
      {/* Any level divs should become components. For now keep here and transfer when planning is done. Also look into using fragments <> and 
      more accessible segments (not divs, section, nav, whatever else) */}
      <div>
        <div>
          <p>
            Im Text that will be updated! Review here!
          </p>
        </div>

        <div>
          <button>1 star</button>
          <button>Text 1</button>
          <button>Text 1</button>
          <button>Text 1</button>
          <button>Text 1</button>
        </div>

        {/* should be same line */}
        <div>
          Your Score: 8
        </div>
        <div>
          AI Score: 12434
        </div>
      </div>
      
    </div>
  );
}

export default App;

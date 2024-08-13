import React, {useState, useEffect} from 'react';
import './styles/App.css'; 
import Header from './components/Header';

function App() {

  const [selectedStars, setSelectedStars] = useState(0)

  const [data, setdata] = useState({
    text: "",
    score: 0,
    pred_score: 0.0,
  });

  //TODO: probably move all calls to API into own file. That can be done later.
  const fetchData = () => {
      // Using fetch to fetch the api from 
      // flask server it will be redirected to proxy
      fetch("/data").then((res) =>
          res.json().then((data) => {
              // Setting a data from api
              setdata({
                  text: data.text,
                  score: data.score,
                  pred_score: data.pred_score,
              });
          })
      );
  };
    
  const cleanText = (input: string) => {
    input = input.replace(/\\"/g, '"')
    return input.replace(/\\n/g, '\n');
  };

  return (
    <div className="App">
      <header>
        {/* HEADER -> Should be some form of nav bar/navigation */}
        
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
          {/* <button onClick={fetchData}>1 star</button>
          <button>Text 1</button>
          <button>Text 1</button>
          <button>Text 1</button>
          <button>Text 1</button> */}
          {/* {<StarCounter/>} */}
        </div>

        <div>
          Your Score: 8
        </div>
        <div>
          AI Score: 12434
        </div>
      </div>

      <div>
        <p>{cleanText(data.text)}</p>
        <p>{data.score}</p>
        <p>{data.pred_score}</p>
      </div>
      
      <button onClick={fetchData}>Submit/Get Results</button>

    </div>
  );
}

export default App;

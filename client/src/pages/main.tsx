import React, {useState, useEffect} from 'react';
import '../styles/App.css'; 
import Header from '../components/Header';
import StarCounter from '../components/StarInput';

function Main() {

  const [data, setdata] = useState({
    text: "",
    score: 0,
    pred_score: 0.0,
  });

  const [scoreModel, setScoreModel] = useState(0)
  const [scoreUser, setScoreUser] = useState(0)
  const [selectedStars, setSelectedStars] = useState(0)


  //TODO: probably move all calls to API into own file. That can be done later.
  const fetchData = () => {
      // Using fetch to fetch the api from 
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
  
  //On load, fetchData. Since there are no dependencies, this will only run once.
  useEffect(
    fetchData,
    []
  )

  // Removed certain strings from input: "\n" and ' \" '
  const cleanText = (input: string) => {
    input = input.replace(/\\"/g, '"')
    return input.replace(/\\n/g, '\n');
  };

  // Calculates the user/model score based on how far off they are from the real score/label
  const calcScore = (realScore: number, pred_score: number, userGuess: number) =>  {
    let modelDiff = Math.abs(realScore - pred_score)
    let userDiff = Math.abs(realScore - userGuess)

    if (userDiff === 0){
      setScoreUser(scoreUser => scoreUser + 2)
    } else if(userDiff <= 1){
      setScoreUser(scoreUser => scoreUser + 1)
    }

    if (modelDiff <= 0.333){
      setScoreModel(scoreModel => scoreModel + 2)
    } else if(modelDiff <= 1){
      setScoreModel(scoreModel => scoreModel + 1)
    }
  }

  return (
    <div className="App">
      <header>
        {/* HEADER -> Should be some form of nav bar/navigation */}
        
        <a href='/info'>Links here</a>
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

        <StarCounter
          selectedStars={selectedStars}
          setSelectedStars={setSelectedStars}
        />

        <div>
          Your Score: {scoreUser}
        </div>
        <div>
          AI Score: {scoreModel}
        </div>
      </div>

      <div>
        <p>{cleanText(data.text)}</p>
        <p>{data.score}</p>
        <p>{data.pred_score}</p>
      </div>
      
      {/*TODO: Either disable buttons or hide one button at a time */}
      <button onClick={() => calcScore(data.score, data.pred_score, selectedStars)}>Submit user Input</button>
      <button onClick={fetchData}>Start/Next Prompt</button>

    </div>
  );
}

export default Main;

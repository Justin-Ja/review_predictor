import React from 'react';
import Header from '../components/Header';
import Title from '../components/Title';

function Info() {
  return (
    <div className="App">
      <Header/>
      <Title text={"Info"}/>
      Important information about the game and how to play it. This is a placeholder for the info page. 
      <>
        <p>:P</p>
      </>
    </div>
  );
}

export default Info;
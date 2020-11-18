import './App.css';
import { jsx, css } from "@emotion/core"
/** @jsx jsx */
/** @jsxRuntime classic */

import { POST, GET, makeid } from './utils';
import React, { useState, useEffect } from 'react';
// import { Button } from '@material-ui/core';


const style = css `
  .imgbox {
    height: 600px;
    width: 900px;
    margin: auto;
    text-align: center;
    background-color: lightgrey;
    overflow: hidden;
    img {
      min-height: 100%;
      min-width: 100%;
    }
  }

  .buttons {
    display: flex;
    justify-content: space-around;
    width: 600px;
    margin: auto;
    padding: 20px;
  }

  .img-tiles {
    display: flex;
    .img {
      height: 200px;
      width: 200px;
      margin: 10px; 
    }
  }

  .navbar {
    display: flex;
    justify-content: space-around;
    padding: 20px;
    margin: 10px;
    border-radius: 4px;
  }

  .header {    
    color: Teal;
    background-color: lightgrey;
    height: 120px;
    text-align: center;
    font-size: 4vw;
    text-transform: capitalize;
  }

`
// console.log(srcs)
const HeaderContent = () => {
  return (
    <div css={style}>
      <div className='header'>
        Property Search Engine
        </div> 
    </div>
  );
}


const PropertyImages = () => {
  const [idx, setIdx] = useState(0)
  const [session_id, setSession_id] = useState(makeid())
  const [srcs, setSrcs] = useState([])
  
  useEffect(()=>GET('users/images').then((r)=>{
    console.log(typeof(r.body))
    setSrcs(r.body)
  }), [])

  console.log(session_id)

  const dislike = () => {
    setIdx((idx+1) % srcs.length);
    const body = {
      session_id: session_id,
      img_id: srcs[idx],
      like: false
    }
    console.log('Dislike')
    POST('users/likes', body)
  }
  
  const like = () => {  
    setIdx((idx+1) % srcs.length);
    const body = {
      session_id: session_id,
      img_id: srcs[idx],
      like: true
    }
    console.log('Like')
    POST('users/likes', body)
  }
  
  return (
  <div css={style}>
    <div className='imgbox'>
      <img src={srcs[idx]} alt='property'/>
    </div>
    <div className='buttons'>
      <button onClick={dislike}>
        Dislike
      </button>
      <button onClick={like}>
        Like
      </button>

    </div>

  </div>
  );
}

const Navbar = () => {
  return (
    <div css={style}>
      <div className='navbar'>
      
      </div>
    </div>
  )
}

const App = () => {
  return (
    <React.Fragment>
      <HeaderContent/>
      <Navbar/>
      <PropertyImages/>
    </React.Fragment>
  )
}

export default App;

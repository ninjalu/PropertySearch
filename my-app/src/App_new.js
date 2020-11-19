import './App.css';
import { jsx, css } from "@emotion/core"
/** @jsx jsx */
/** @jsxRuntime classic */

import { POST } from './utils';
import React, { useState } from 'react';
import { Button } from '@material-ui/core';


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
// const srcs = ['https://livforcake.com/wp-content/uploads/2017/06/vanilla-cake-thumb-500x500.jpg',
// 'https://chelsweets.com/wp-content/uploads/2019/04/IMG_1029-2-735x1103.jpg', 
// 'https://img.taste.com.au/DqTMY6Cz/taste/2018/08/smarties-chocolate-cake-139872-2.jpg']

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

  // const [src, setSrc] = useState(0)

  const dislike = () => {
    const body = {
      session_id: 'we',
      img_id: 'wjo',
      like: false
    }
    console.log('Dislike')
    // setSrc((src+1) % srcs.length)
    POST('user/likes', body)
  }
  
  const like = () => {
    const body = {
      session_id: 'we',
      img_id: 'wjo',
      like: true
    }
    console.log('Like')
    // setSrc((src+1) % srcs.length)
    POST('user/likes', body)
  }
  
  return (
  <div css={style}>
    <div className='imgbox'>
      <img src='https://img.taste.com.au/DqTMY6Cz/taste/2018/08/smarties-chocolate-cake-139872-2.jpg' alt='property'/>
    </div>
    <div className='buttons'>
      <Button onClick={dislike}>
        Dislike
      </Button>
      <Button onClick={like}>
        Like
      </Button>

    </div>

  </div>
  );
}

const Navbar = () => {
  return (
    <div css={style}>
      <div className='navbar'>
        nav bar goes here
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

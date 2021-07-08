import './App.css';

import React, { useState } from 'react';
import { Redirect, Route, BrowserRouter as Router, Switch } from 'react-router-dom';

import Home from './pages/home';

// import Login from './pages/login';

function App() {
  const [isAuth, setIsAuth] = useState(true);
  const reactAppRouteUrl = process.env.REACT_APP_ROUTE_URL || '/';

  return (
    <div className="App">
      <Router>
        <Switch>
          {/*<Route path="/login">*/}
          {/*  {isAuth ? <Redirect to="/" /> : <Login isAuth={isAuth} setIsAuth={setIsAuth} />}*/}
          {/*</Route>*/}
          <Route exact path={reactAppRouteUrl}>
            <Home isAuth={isAuth} setIsAuth={setIsAuth} />
          </Route>
        </Switch>
      </Router>
    </div>
  );
}

export default App;

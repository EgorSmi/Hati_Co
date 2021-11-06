import "./styles/main.sass"
import 'bootstrap/dist/css/bootstrap.min.css'

import {useEffect, useState} from "react"

import { Switch, Route, useLocation} from "react-router-dom";

import PAGES, {PAGES_COMPONENTS} from "./pages"

import { Footer } from "./components/footer/Footer";

function App() {

  const [activePage, setActivePage] = useState(null)
  let location = useLocation()

  useEffect(() => {
    const page = PAGES.find((el) => el.path === location.pathname)
    if(!page){
    }else{
      setActivePage(page)
    }
  }, [location]);

  return (
    <div className="App">
      <Switch>
        {PAGES.map((el, index) => {
            return (
                <Route key={index} path={el.path} exact>
                    {PAGES_COMPONENTS[el.name]}
                </Route>
            );
        })}
      </Switch>
      <Footer/>
    </div>
  );
}

export default App;

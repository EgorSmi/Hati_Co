// import { createStore, applyMiddleware, compose } from "redux";
// import thunk from "redux-thunk";
// import rootReducer from "../reducers"

// const initialState = {};

// const middleware = [thunk];

// const store = createStore(
//   rootReducer,
//   initialState,
//   compose(
//     applyMiddleware(...middleware),
//     (window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ &&
//       window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__()) ||
//       compose
//   )
// );

// export default store;


import {combineReducers, configureStore} from "@reduxjs/toolkit"

import MetaSlice from "../reducers/MetaSlice"

const rootReducer = combineReducers({
  MetaSlice
})

export const setupStore = () => {
    return configureStore({
      reducer: rootReducer
    }) 
}
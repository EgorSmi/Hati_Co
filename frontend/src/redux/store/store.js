import {
  combineReducers, 
  configureStore
} from "@reduxjs/toolkit"

import reducers from "../reducers"

const rootReducer = combineReducers({
  reducers
})

export const setupStore = () => {
    return configureStore({
      reducer: rootReducer
    }) 
}
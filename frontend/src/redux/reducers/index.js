import { combineReducers } from "redux"

// async query
import meta from "./modules/meta/MetaSlice"
import photoFilterSlice from "./modules/photo/PhotoFilterSlice"


export default combineReducers({
    meta,
    photoFilterSlice,
})
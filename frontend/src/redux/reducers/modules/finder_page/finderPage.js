import {
    SET_ISTEST_DATA
} from "./types"

const initialState = {
    is_testdata: false
}


export default function reducer(state = initialState, action){
    switch(action.type){
        case SET_ISTEST_DATA:
            return{
                ...state,
                is_testdata: action.payload
            }
        default: 
            return state 
    }
}
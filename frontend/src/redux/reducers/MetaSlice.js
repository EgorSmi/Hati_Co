import {createSlice} from "@reduxjs/toolkit"
import { fetchMeta } from "./actionCreator"

const initialState = {
    meta: {},
    isLoading: true,
    error: ''
}

export const metaSlice = createSlice({
    name: 'meta',
    initialState,
    reducers: {},
    extraReducers: {
        [fetchMeta.fulfilled.type]: (state, action) => {
            state.isLoading = false;
            state.error = ''
            state.meta = action.payload
        },
        [fetchMeta.pending.type]: (state, action) => {
            state.isLoading = true
        },
        [fetchMeta.rejected.type]: (state, action) => {
            state.isLoading = false
            state.error = action.payload
        }
    }
})

export default metaSlice.reducer
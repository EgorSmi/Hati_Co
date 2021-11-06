import {createSlice} from "@reduxjs/toolkit"
import { fetchMeta } from "../../asyncActionCreator"

const initialState = {
    meta: {},
    isLoadingMeta: true,
    error: ''
}

export const metaSlice = createSlice({
    name: 'meta',
    initialState,
    reducers: {},
    extraReducers: {
        [fetchMeta.fulfilled.type]: (state, action) => {
            state.isLoadingMeta = false;
            state.error = ''
            state.meta = action.payload
        },
        [fetchMeta.pending.type]: (state, action) => {
            state.isLoadingMeta = true
        },
        [fetchMeta.rejected.type]: (state, action) => {
            state.isLoadingMeta = false
            state.error = action.payload
        }
    }
})

export default metaSlice.reducer
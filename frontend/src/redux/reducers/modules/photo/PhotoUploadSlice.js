import {createSlice} from "@reduxjs/toolkit"
import { uploadPhoto } from "../../asyncActionCreator"

const initialState = {
    isLoadingPhoto: null,
    error: '',
    id_object: null
}

export const photoUploadSlice = createSlice({
    name: 'photo_upload',
    initialState,
    reducers: {},
    extraReducers: {
        [uploadPhoto.fulfilled.type]: (state, action) => {
            state.isLoadingPhoto = false;
            state.error = ''

            if(action.payload['id']){
                state.id_object = action.payload
            }else{
                state.id_object = undefined
            }
        },
        [uploadPhoto.pending.type]: (state, action) => {
            state.isLoadingPhoto = true
        },
        [uploadPhoto.rejected.type]: (state, action) => {
            state.isLoadingPhoto = false
            state.error = action.payload
        }
    }
})

export default photoUploadSlice.reducer
import {createSlice} from "@reduxjs/toolkit"
import {filterPhotoUpload} from "../../asyncActionCreator"

const initialState = {
    filterPhoto: {},
    isLoadingFilter: true,
    errorFilter_msg: '',
    errorFilter_status: ''
}

export const photoFilterSlice = createSlice({
    name: 'photo_filter',
    initialState,
    reducers: {},
    extraReducers: {
        [filterPhotoUpload.fulfilled.type]: (state, action) => {
            state.isLoadingFilter = false;
            state.errorFilter_status = ''
            state.filterPhoto = action.payload
            state.errorFilter_status = 'succeeded'
        },
        [filterPhotoUpload.pending.type]: (state, action) => {
            state.isLoadingFilter = true
            state.errorFilter_status = 'loading'
        },
        [filterPhotoUpload.rejected.type]: (state, action) => {
            state.isLoadingFilter = false
            state.errorFilter_msg = action.payload
            state.errorFilter_status = 'rejected'
        }
    }
})

export default photoFilterSlice.reducer
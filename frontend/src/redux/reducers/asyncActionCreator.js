import axios from "axios"
import {createAsyncThunk} from "@reduxjs/toolkit";


export const fetchMeta = createAsyncThunk(
    'api/hati/meta/',
    async thunkAPI => {
        try {
            const response = await axios.get('http://84.201.148.17:26555/api/hati/meta')
            // const response = await axios.get('http://127.0.0.1:8000/api/hati/meta')
            return response.data
        } catch(e){
            return thunkAPI.rejectWithValue("Не удалось загрузить meta данне")
        }
    }
)

export const filterPhotoUpload = createAsyncThunk(
    '/api/hati/advertisement/',
    async(filterData, thunkAPI) => {

        let {
            imagedata,
            selectedColorId,
            selectedAnimal,
            selectedBreed,
            selectedTail,
            selectWithowner,
            selectMarks,
            selectIsDog,
            radius,
            cumId,
            is_testdata
        } = filterData

        let bodyFormData = new FormData()

        if(is_testdata === "false"){
            for(let i = 0; i < imagedata.length; i++ ){
                bodyFormData.set(`image_${i}`, imagedata[i])
            }
        }

        bodyFormData.append('radius', radius)
        bodyFormData.append('camera', cumId)
        bodyFormData.append('breed', selectedBreed?.value)
        bodyFormData.append('animal', selectedAnimal?.value)
        bodyFormData.append('tail', selectedTail?.value)
        bodyFormData.append('color', selectedColorId?.value)
        bodyFormData.append('withowner', selectWithowner?.value)
        bodyFormData.append('markers', selectMarks?.value)
        bodyFormData.append('isitadog', selectIsDog?.value)
        bodyFormData.append('is_testdata', is_testdata)

        try{
            const response = await axios.post('http://84.201.148.17:26555/api/hati/advertisement/', bodyFormData)
            // const response = await axios.post('http://127.0.0.1:8000/api/hati/advertisement/', bodyFormData)
            return response.data.images
        } catch(e){
            return thunkAPI.rejectWithValue("Что то пошло не так... попробуйте еще")
        }
    }
)

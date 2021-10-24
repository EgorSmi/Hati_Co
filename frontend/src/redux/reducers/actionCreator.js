import axios from "axios"
import {createAsyncThunk} from "@reduxjs/toolkit";

export const fetchMeta = createAsyncThunk(
    'api/hati/meta/',
    async thunkAPI => {
        try {
            const response = await axios.get('http://127.0.0.1:8000/api/hati/meta')
            return response.data
        } catch(e){
            return thunkAPI.rejectWithValue("Не удалось загрузить meta данне")
        }
    }
)


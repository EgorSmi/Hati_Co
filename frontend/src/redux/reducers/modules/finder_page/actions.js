import{
    SET_ISTEST_DATA
} from "./types"

export const testDataToggle = is_testdata => {
    return{
        type: SET_ISTEST_DATA,
        payload: is_testdata
    }
}
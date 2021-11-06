import "./styles/main.sass"
import {Row, Col} from "react-bootstrap"
import {useEffect, useState } from "react"

import ImageUploader from 'react-images-upload'
import {FilterButton} from "../../../../components/button/FilterButton"
import Loader from "../../../../components/loader/Loader"

import {useDispatch, useSelector} from "react-redux"
import {filterPhotoUpload} from "../../../../redux/reducers/asyncActionCreator"

export function ImgsUploader({
    filterData, 
}){
    const dispatch = useDispatch()

    const [pictures, setPictures] = useState([])

    const {
        isLoadingFilter, 
        errorFilter_status, errorFilter_msg
    } = useSelector(state => state.reducers.photoFilterSlice)

    const onDrop = picture => {
        setPictures(picture)
    }

    const submitHandler = () => {
        /* upload imgs submit button */

        if(pictures.length === 0){
            filterData['is_testdata'] = "true"
            dispatch(
                filterPhotoUpload(
                    filterData
                )
            )
        }else{
            let imagedata = document.querySelector('input[type="file"]').files
            filterData['is_testdata'] = "false"
            filterData['imagedata'] = imagedata
            dispatch(
                filterPhotoUpload(
                    filterData
                )
            )
        }
    }

    return(
        <Row>
            <Col xs={12} className={"image_uploader"}>
                <ImageUploader
                    singleImage={false}
                    withIcon={true}
                    multiple={true}
                    withPreview={true}
                    buttonText='Выберите фотографии питомца'
                    onChange={onDrop}
                    imgExtension={['.jpg','.png','.jpeg']}
                    maxFileSize={3145728}
                />
                <p>Не загружайте фото, если хотите использовать тестовый датасет</p>
                <Row className={"filter_detail_box pt-3"}>
                    <Col xs={6}>
                        {errorFilter_status === 'loading' && 
                            <Loader/>
                        }
                        {
                            errorFilter_status === 'rejected' &&
                            <Row className={"animal_undefined"}>
                                <Col xs={8}>
                                    <p>
                                        {errorFilter_msg}
                                    </p>
                                </Col>
                            </Row> 
                        }
                    </Col>
                    <Col xs={3} className={"m-auto ml-5"}>
                    </Col>
                    <Col xs={3} className={"m-auto"}>
                        <FilterButton
                            onClick={submitHandler}
                        >
                            Найти питомца:)
                        </FilterButton>
                    </Col>
                </Row>
            </Col>
        </Row>
    )
}
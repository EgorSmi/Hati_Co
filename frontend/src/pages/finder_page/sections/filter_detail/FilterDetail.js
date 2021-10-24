import "./styles/main.sass"

import {Col, Row} from "react-bootstrap"

import SelectMain from "../../../../components/selects/Select"
import { toSelectData } from "../../../../components/selects/dataHandler/toSelectData"
import {SliderInfo} from "../../sections/slider_info/SliderInfo"

import { GithubPicker } from 'react-color'

import {FinderMap} from "../finder_map/FinderMap"

import { FancyBoxSlider } from "../../../../components/fancybox-slider/FancyBoxSlider"

import axios from "axios"

import {useState} from "react"

import ImageUploader from 'react-images-upload';


// let sliderDataTest = [
//     {
//         image: "https://images.unsplash.com/photo-1618335829737-2228915674e0?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=750&q=80",
//         description: "Ул Первомайская дом 2, кв 21"
//     },
//     {
//         image: "https://images.unsplash.com/photo-1601458457252-9d1898a84c93?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=750&q=80",
//         description: "Ул Первомайская дом 2, кв 25"
//     },
//     {
//         image: "https://images.unsplash.com/photo-1544256718-3bcf237f3974?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=751&q=80",
//         description: "Ул Первомайская дом 2, кв 30"
//     },
//     {
//         image: "https://images.unsplash.com/photo-1549605659-32d82da3a059?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=750&q=80",
//         description: "Ул Первомайская дом 2, кв 35"
//     }
// ]


export function FilterDetail({meta, isLoading}){

    
    const [pictures, setPictures] = useState([])
    const [selectedColorId, setSelectedColorId] = useState([])
    const [selectedAnimal, setSelectedAnimal] = useState([])
    const [selectedBreed, setSelectedBreed] = useState([])
    const [selectedTail, setSelectedTail] = useState([])

    const [cumId, setCumId] = useState([])
    
    let breed, animal, tail, colors, colorId

    if(isLoading === false){
        breed = toSelectData(meta.breed, 'id', 'name')
        animal = toSelectData(meta.animal, 'id', 'name')
        tail = toSelectData(meta.tail, 'id', 'name')
        colors = toSelectData(meta.color, 'id', 'name')
    }

    const onDrop = picture => {
        setPictures([...pictures, picture])
    }

    const submitHandler = () => {
        var imagedata = document.querySelector('input[type="file"]')
      
            var bodyFormData = new FormData();
            var imagedata = document.querySelector('input[type="file"]')

            bodyFormData.append('camera', cumId)
            bodyFormData.append('animal', selectedAnimal.value)
            bodyFormData.append('tail', selectedTail.value)
            bodyFormData.append('color', selectedColorId.value)
            bodyFormData.append('image', imagedata.files[0])
            bodyFormData.append('breed', selectedBreed.value)

            axios({
                method: "post",
                url: "http://127.0.0.1:8000/api/hati/advertisement/",
                data: bodyFormData,
                headers: { "Content-Type": "multipart/form-data" },
            })
            .then(function (response) {
                console.log(response);
            })
            .catch(function (response) {
                console.log(response);
            });
        }
    

    return(
        <Row className="filter_detail">
            <Row>
                <Col xs={6}>
                    <SelectMain
                        isDisabled={isLoading}
                        isLoading={isLoading}
                        onChange={setSelectedAnimal}
                        selectData={animal}
                        placeholder={"Животное"}
                    />
                </Col>
                <Col xs={6}>
                    <SelectMain
                        isDisabled={isLoading}
                        isLoading={isLoading}
                        onChange={setSelectedBreed}
                        selectData={breed}
                        placeholder={"Порода собаки"}
                    />
                </Col>
            </Row>
            <Row className="pt-3">
                <Col xs={6}>
                    {/* <p>Выберите цвет:</p> */}
                    {/* <GithubPicker
                        colors={colors}
                        onChange={colorHandler}
                    /> */}
                    <SelectMain
                        isDisabled={isLoading}
                        isLoading={isLoading}
                        onChange={setSelectedColorId}
                        selectData={colors}
                        placeholder={"Цвет собаки"}
                    />
                </Col>
                <Col xs={6}>
                    <SelectMain
                        isDisabled={isLoading}
                        isLoading={isLoading}
                        onChange={setSelectedTail}
                        selectData={tail}
                        placeholder={"Длина хвоста"}
                    />
                </Col>
            </Row>
            <Row>
                <Col xs={12} className="pt-3">
                    <p>Загрузите фото:</p>
                    <ImageUploader
                        singleImage={true}
                        withIcon={true}
                        withPreview={true}
                        buttonText='Выберите фото собаки'
                        onChange={onDrop}
                        imgExtension={['.jpg','.png', '.jpeg']}
                        maxFileSize={5242880}
                    />
                </Col>
            </Row>
            <Row>
                <Col xs={12}>
                    <p>Выберите камеру: {cumId}</p>
                    <FinderMap
                        setCumId={setCumId}
                    />
                    <Row className={"filter_detail_box"}>
                        <button 
                            className={"filter_detail__button"}
                            onClick={submitHandler}
                        >
                            Найти питомца:)
                        </button>
                    </Row>
                    {/* <FancyBoxSlider
                        sliderData={sliderDataTest}
                        isLoading={isLoading}
                    /> */}
                </Col>
            </Row>
        </Row>
    )
}
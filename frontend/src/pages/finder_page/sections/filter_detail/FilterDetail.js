import "./styles/main.sass"

import {Col, Row} from "react-bootstrap"

import {useState, Fragment, useEffect} from "react"

import useInput from "../../../../hooks/useInput"

import { toSelectData } from "../../../../components/selects/dataHandler/toSelectData"
import FilterNumInput from "../../../../components/radiusInput/RadiusInput"
import SelectMain from "../../../../components/selects/Select"

import { FinderMap } from "./finder_map/FinderMap"

import {ImgsUploader} from "../imgs_uploader/ImgsUploader"

import {useDispatch, useSelector} from "react-redux"
import { fetchMeta } from "../../../../redux/reducers/asyncActionCreator"

import { 
    with_owner, 
    markers,
    is_animal,
    is_dog
} from "./selectData"

export function FilterDetail({
    setStage, 
}){
  
    const dispatch = useDispatch()

    // loading metaData to selects
    const {meta, isLoadingMeta} = useSelector(state => state.reducers.meta)
    useEffect(() => {
        dispatch(fetchMeta())
    }, [])

    const [selectedColorId, setSelectedColorId] = useState('undefined')
    const [selectedAnimal, setSelectedAnimal] = useState('undefined')
    const [selectedBreed, setSelectedBreed] = useState('undefined')
    const [selectedTail, setSelectedTail] = useState('undefined')

    const [selectWithowner, setSelectWithowner] = useState('undefined')
    const [selectMarks, setSelectMarks] = useState('undefined')
    const [selectIsDog, setSelectIsDog] = useState('undefined')

    const [radius, bindRadius, resetRadius] = useInput('undefined')
    const [cumId, setCumId] = useState('')

    let breed, animal, tail, colors;

    if(isLoadingMeta === false){
        breed = toSelectData(meta.breed, 'id', 'name')
        animal = toSelectData(meta.animal, 'id', 'name')
        tail = toSelectData(meta.tail, 'id', 'name')
        colors = toSelectData(meta.color, 'id', 'name')
    }

    let filterData = {
        "selectedColorId": selectedColorId,
        "selectedAnimal": selectedAnimal,
        "selectedBreed": selectedBreed,
        "selectedTail": selectedTail,
        "selectWithowner": selectWithowner,
        "selectMarks": selectMarks,
        "selectIsDog": selectIsDog,
        "radius": radius,
        "cumId": cumId,
    }

    return(
        <Fragment>
            <Row>
                <p>Фотографии питомца найдены, заполните фильтры </p>
            </Row>
            <Row>
                <Col xs={6}>
                    <SelectMain
                        isDisabled={isLoadingMeta}
                        isLoading={isLoadingMeta}
                        onChange={setSelectedAnimal}
                        selectData={is_animal}
                        placeholder={"Есть ли животное"}
                    />
                </Col>
                <Col xs={6}>
                    <SelectMain
                        isDisabled={isLoadingMeta}
                        isLoading={isLoadingMeta}
                        onChange={setSelectedBreed}
                        selectData={breed}
                        placeholder={"Порода питомца"}
                    />
                </Col>
            </Row>
            <Row className={"pt-3"}>
                <Col xs={6}>
                    <SelectMain
                        isDisabled={isLoadingMeta}
                        isLoading={isLoadingMeta}
                        onChange={setSelectedColorId}
                        selectData={colors}
                        placeholder={"Цвет питомца"}
                    />
                </Col>
                <Col xs={6}>
                    <SelectMain
                        isDisabled={isLoadingMeta}
                        isLoading={isLoadingMeta}
                        onChange={setSelectedTail}
                        selectData={tail}
                        placeholder={"Длина хвоста"}
                    />
                </Col>
            </Row>
            <Row className={"pt-3 align-items-center"}>
                <Col xs={4}>
                    <SelectMain
                        isDisabled={isLoadingMeta}
                        isLoading={isLoadingMeta}
                        onChange={setSelectWithowner}
                        selectData={with_owner}
                        placeholder={"Питомец с хозяином?"}
                    />
                </Col>
                <Col xs={4}>
                    <SelectMain
                        isDisabled={isLoadingMeta}
                        isLoading={isLoadingMeta}
                        onChange={setSelectIsDog}
                        selectData={is_dog}
                        placeholder={"Есть ли собака?"}
                    />
                </Col>
                <Col xs={4}>
                    <SelectMain
                        isDisabled={isLoadingMeta}
                        isLoading={isLoadingMeta}
                        onChange={setSelectMarks}
                        selectData={markers}
                        placeholder={"Наличие атрибутов домашней собаки"}
                    />
                </Col>
            </Row>
            <Row className={"text-align-center"}>
                <Col xs={6} className={"pt-3 m-auto"}>
                    <p>Выберите камеру: {cumId}</p>
                </Col>
                <Col xs={6} className={"pt-3"}>
                    <FilterNumInput
                        bind={bindRadius}
                        placeholder={"Радиус поиска камеры"}
                    />
                </Col>
            </Row>
            <Row>
                <Col xs={12} className={"pt-3"}>
                    <FinderMap
                        setCumId={setCumId}
                    />
                </Col>
            </Row>
                <ImgsUploader
                    setStage={setStage}
                    filterData={filterData}
                />
        </Fragment>
    )
}
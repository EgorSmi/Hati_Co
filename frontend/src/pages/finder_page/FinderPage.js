import "./styles/main.sass"

import {Col, Row} from "react-bootstrap"
import {Section} from "../../components/section"

import {FilterDetail} from "./sections/filter_detail/FilterDetail"

import {useEffect, useState} from "react"

import {useDispatch, useSelector} from "react-redux"

import {SliderInfo} from "./sections/slider_info/SliderInfo"

export function FinderPage(){

    const [stage, setStage] = useState(1) 

    const {
        filterPhoto, isLoadingFilter, 
        errorFilter_status, errorFilter_msg
    } = useSelector(state => state.reducers.photoFilterSlice)



    useEffect(() => {
        if(errorFilter_status === 'succeeded'){
            setStage(2)
        }
    }, [errorFilter_status])
   

    return(
        <Section className={"finder"}>
            <Row className={"finder__wrapper"}>
                <Col xs={12} lg={8}>
                    <Row className={"finder__title"}>
                        Сервис для поиска пропавших животных
                    </Row>
                    <Row className={"finder__content"}>
                            {stage !== 2 &&
                                <FilterDetail/> 
                            }
                            {stage === 2 &&
                                <SliderInfo
                                    setStage={setStage}
                                    filterPhoto={filterPhoto}
                                    error_status={errorFilter_status}
                                    error_msg={errorFilter_msg}
                                />
                            }
                    </Row>
                </Col>
            </Row>
        </Section>
    )
}
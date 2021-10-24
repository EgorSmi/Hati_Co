import "./styles/main.sass"

import {Col, Row} from "react-bootstrap"
import { Section } from "../../components/section"

import {FilterDetail} from "./sections/filter_detail/FilterDetail"
import {SliderInfo} from "./sections/slider_info/SliderInfo"
import {FinderMap} from "./sections/finder_map/FinderMap"

import {useEffect} from "react"

import {fetchMeta} from "../../redux/reducers/actionCreator"
import {useDispatch} from "react-redux";
import {useSelector} from "react-redux"

export function FinderPage(){

    const dispatch = useDispatch()

    const meta = useSelector(state => state.MetaSlice.meta)
    const isLoading = useSelector(state => state.MetaSlice.isLoading)

    useEffect( () => {
        dispatch(fetchMeta())
    }, [])

    return(
        <Section className={"finder"}>
            <Row className={"finder__wrapper"}>
                <Col xs={8}>
                    <FilterDetail 
                        meta={meta}
                        isLoading={isLoading}
                    />
                </Col>
            </Row>
        </Section>
    )
}
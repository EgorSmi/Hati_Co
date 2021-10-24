import { Row } from "react-bootstrap"

import { FancyBoxSlider } from "../../../../components/fancybox-slider/FancyBoxSlider"


export function SliderInfo(props){
    console.log(props.sliderInfo)
    return(
        <Row>
            <FancyBoxSlider
                sliderInfo={props.sliderInfo}
                isLoading={props.isLoading}
            />
        </Row>
    )
}
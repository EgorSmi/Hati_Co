import {Row, Col} from "react-bootstrap"

import {FancyBoxSlider} from "../../../../components/fancybox-slider/FancyBoxSlider"
import {FilterButton} from "../../../../components/button/FilterButton"


export function SliderInfo({filterPhoto}){

    const length = filterPhoto.length

    const refreshPage = () => {
        window.location.reload()
    }

    return(
        <Row>
            <FancyBoxSlider
                filterPhoto={filterPhoto}
            />
            <Row>
                {
                    length === 0  &&
                    <Row className={"animal_undefined"}>
                        <Col xs={8}>
                            <p>
                                Питомец не найден :( , попробуйте еще
                            </p>
                        </Col>
                        <Col xs={3}>
                            <FilterButton
                                onClick={refreshPage}
                            >
                                Попробовать снова?
                            </FilterButton>
                        </Col>
                    </Row> 
                }
            </Row>
        </Row>
    )
} 
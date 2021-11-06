import "./styles/main.sass"

import { ReactSVG } from 'react-svg'
import  arrow  from "../../assets/icons/arrow.svg"

import {useState, Fragment} from "react"


export function FancyBoxSlider({filterPhoto}){

    const [current, setCurrent] = useState(0)
    const length = filterPhoto.length - 1

    const handleClickNext = () => {
        if(current >= 0 && current < length){
            setCurrent(current + 1)
        }else{
            setCurrent(0)
        }
    }

    const handleClickPrev = () => {
        if(current <= length && current > 0){
            setCurrent(current - 1)
        }else{
            setCurrent(length)
        }
    }

    const handler = (src) => {
        window.open(`${src}`)
    }

    return (
        <div className={"fancybox"}>
            {  length >= 0 &&
                <Fragment>
                        <div className={"fancybox__navigation"}>
                            <button 
                                data-fancybox-prev 
                                className={"fancybox-button fancybox-button__arrow_left"}
                                onClick={handleClickPrev}
                            >
                                <ReactSVG src={arrow}/>
                            </button>

                            <button 
                                data-fancybox-next 
                                className={"fancybox-button fancybox-button__arrow_right"}
                                onClick={handleClickNext}
                            >
                                <ReactSVG src={arrow}/>
                            </button>
                        </div>
                        <div className={"fancybox-content"}>
                            {filterPhoto.map((slide, index) => {
                                return (
                                    <div key={slide.id} >
                                        {index === current && 
                                            <Fragment>
                                                <img 
                                                    src={slide.image} 
                                                    onClick={() => handler(slide.image) }
                                                    className={"fancybox-image"}
                                                />
                                                <p>{slide.address}</p>
                                            </Fragment>
                                        }
                                    </div>
                                )
                            })}
                        </div>
                </Fragment>
            }
        </div>
    )
}
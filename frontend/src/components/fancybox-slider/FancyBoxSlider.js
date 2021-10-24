import "./styles/main.sass"

import { ReactSVG } from 'react-svg'
import  arrow  from "../../assets/icons/arrow.svg"

import {useState} from "react";

import { Fragment } from "react";

export function FancyBoxSlider({sliderData, isLoading}){

    const [current, setCurrent] = useState(0);
    const length = sliderData.length - 1

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

    return (
        <div className={"fancybox"}>
            {isLoading === false &&
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
                        {sliderData.map((slide, index) => {
                            return (
                                <div>
                                    {index === current && 
                                    <div>
                                        <img 
                                        key={index} 
                                        src={slide.image} 
                                        className={"fancybox-image"}
                                        />
                                        <p>{slide.description}</p>
                                    </div>
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
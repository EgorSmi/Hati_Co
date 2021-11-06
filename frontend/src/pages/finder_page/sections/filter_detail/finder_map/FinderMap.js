import "./styles/main.sass"

import axios from "axios"
import {useState, useEffect} from "react"


const Iframe = ({source, isLoading, setCumId}) => {

    let iframe, iframeDoc, dataCams, idCam;

    useEffect(() => {
        document.getElementsByTagName('iframe')[0].onload = function() {
            iframe = document.getElementsByTagName('iframe')[0]
            let iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
            iframeDoc.onclick = function(){
                const timer2 = setTimeout(() => {
                    if(iframeDoc.querySelector(".leaflet-popup-content div") != null){
                        setCumId(iframeDoc.querySelector(".leaflet-popup-content div").textContent)
                    }
                }, 50)
            }
        }

    }, [source, isLoading]);

    return(
        <iframe className={"map_frame"} srcDoc={source}></iframe>
    )
}

export function FinderMap(props){

    const [html, setHtml] = useState('')
    const [isLoading, setIsLoading] = useState(true)

    // axios.get('http://127.0.0.1:8000/api/hati/map').then((resp) => {
    axios.get('http://84.201.148.17:26555/api/hati/map').then((resp) => {
        setHtml(resp.data)
        setIsLoading(false)
    })

    return(
        <div>
            <Iframe 
            source={html} 
            isLoading={isLoading}
            setCumId={props.setCumId}
            />
        </div>
    )
}
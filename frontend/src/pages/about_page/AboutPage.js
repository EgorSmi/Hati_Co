import { Section } from "../../components/section"
import {Row} from "react-bootstrap"

export function AboutPage(){
    return(
        <Section className={"about"}>
            <Row>
                <h1>О нас</h1>
            </Row>
            <Row className={"about__wrapper"}>
                <iframe 
                    src="https://docs.google.com/presentation/d/e/2PACX-1vSzIR9_KAE-p-O9XnEH5HKT0rVy2ghX96oEFuoBV7zC85xJvwb4K9qrsySYioJ0RBrW3wC6mK6eZRIl/embed?start=true&loop=false&delayms=3000" 
                    frameborder="0" 
                    width="960"
                    height="569" 
                    allowfullscreen="true" 
                    mozallowfullscreen="true" 
                    webkitallowfullscreen="true">
                </iframe>
            </Row>
        </Section>
    )
}

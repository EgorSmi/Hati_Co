import "./styles/main.sass"
import { Section } from "../section"

export function Footer(){
    return(
        <footer className={"footer"}>
            <Section>
                <span>Hati.Co</span>
                <a href={"/"}>Главная</a>
                <a href={"/about_us"}>О нас</a>
            </Section>
        </footer>
    )
}
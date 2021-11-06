import {Container} from "react-bootstrap";

export function Section({
    className = "",
    children = undefined,
    sectionName = "",
}) {
    return(
        <Container id={sectionName} className={`section ${className}`} fluid>
            <Container className="section_block">
                {children}
            </Container>
        </Container>
    )
}
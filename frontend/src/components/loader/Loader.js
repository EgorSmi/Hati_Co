import './styles/Loader.css'
import "./styles/main.sass"

import { Row } from 'react-bootstrap'

export default () => (
    <Row className={"finding_box finding_loader relative"}>
        <div className="lds-dual-ring"/>
    </Row>
)


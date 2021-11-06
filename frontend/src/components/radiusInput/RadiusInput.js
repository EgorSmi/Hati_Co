import "./styles/main.sass"

export default function FilterNumInput(props){

    const bind = props.bind
    const placeholder = props.placeholder ?? ''

    return(
        <input
            {...bind}
            placeholder={placeholder}
            type="number"
            className="filter_number__input"
        />
    )
}